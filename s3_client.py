
import argparse
import sys
import os
from datetime import datetime
import logging
from functools import wraps
import hashlib
# from multiprocessing import Pool

from boto3 import client
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s"
)
LOGGER = logging.getLogger()

SUBCOMMANDS = {'ls', 'list', 'put', 'upload', 'get', 'download', 'rm', 'delete', 'mv', 'move', 'md5'}


def _firstSubcommand(argv):
    '''
    Return the index of the first subcommand in argv.
    '''
    for i in range(1, len(argv)):
        if argv[i] in SUBCOMMANDS:
            return i
    return len(argv)


def format_size(size, si=False):
    '''
    Format file size in bytes to human readable format.

    Parameters
    ----------
    size: int
        Size of the file in bytes
    si: bool
        If True, use base 1000 instead of 1024

    Returns
    -------
    formated_size: str
        Formated file size with the appropriate prefix multiplier.
    '''
    divisor = 1000 if si else 1024.0
    for i, unit in enumerate(("", "K", "M", "G", "T", "P")):
        if i > 0:
            size /= divisor
        if abs(size) < divisor:
            size_f = round(size, 1)
            if size_f % 1 == 0 or size_f >= 100:
                size_f = round(size)
            return f"{size_f}{unit.lower() if si else unit}"
    return f'{size:.0e}{"p" if si else "P"}'


def s3_client_function(f):
    '''
    Wraps files that use the boto3 client in a try block to catch boto3.ClientError exception
    '''
    @wraps(f)
    def try_block(*args, **kwargs):
        try:
            return f(*args, **kwargs)

        except ClientError as e:
            LOGGER.error(e)
            sys.exit(1)
    return try_block


def get_s3_file_md5(bucket, s3_client, key, calculate=False, verbose=True):
    """
    Calculate the MD5 hash of a file stored in an Amazon S3 bucket without downloading the whole file.

    Parameters
    ----------
    bucket: str
        The name of the S3 bucket.
    s3_client: boto3.client
        Initialized client object
    key: str
        The key (path) of the file in the S3 bucket.
    calculate: bool
        If the ETag is not a md5 hash should the file be downloaded in pieces to calculate
        the hash manually? Default is False.
    verbse: bool
        Print verbose output? Default is True.

    Returns:
    - The MD5 hash of the file.
    """

    # Get the object metadata, including the ETag which contains the MD5 hash
    response = s3_client.head_object(Bucket=bucket, Key=key)

    # Extract the ETag (which should contain the MD5 hash)
    etag = response['ETag']

    # ETag format might include quotes, remove them if present
    if etag.startswith('"') and etag.endswith('"'):
        etag = etag[1:-1]

    # If the ETag is an MD5 hash (32 hexadecimal characters), return it
    if len(etag) == 32:
        return etag

    if not calculate:
        return None

    # If the ETag is not a standard MD5 hash, compute the MD5 hash manually by downloading parts
    md5 = hashlib.md5()
    size_bytes = response['ContentLength']
    parts = size_bytes // (5 * 1024 * 1024) + 1  # Calculate number of parts (5MB each)

    # Iterate over parts and calculate MD5 hash
    for i in (tqdm(range(parts), desc='Calculating hash', leave=False) if verbose else range(parts)):
        range_start = i * (5 * 1024 * 1024)
        range_end = min((i + 1) * (5 * 1024 * 1024), size_bytes) - 1
        response = s3_client.get_object(Bucket=bucket, Key=key, Range=f'bytes={range_start}-{range_end}')
        md5.update(response['Body'].read())

    return md5.hexdigest()


@s3_client_function
def file_exists(bucket, s3_client, file_path):
    '''
    Check if a file exists in a s3 bucket.

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    file_path: str
        Path on s3 bucket for file to check.

    Returns
    -------
    file_exists: bool
    '''
    kwargs = {'Prefix': file_path}
    response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
    if 'Contents' in response:
        if file_path in [x for x in [file['Key'] for file in response['Contents']]]:
            return True
    while response['IsTruncated']:
        kwargs['ContinuationToken'] = response['NextContinuationToken']
        response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
        if file_path in [x for x in [file['Key'] for file in response['Contents']]]:
            return True
    return False


@s3_client_function
def upload_file(bucket, s3_client, file_to_upload, location, quiet=False):
    '''
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    location: str
        The directory on s3 to upload to
    file_name: str
        File to upload
    '''
    _location = f'{location.rstrip("/")}/{os.path.basename(file_to_upload)}'
    GB = 1024 ** 3
    config=TransferConfig(multipart_threshold=5*GB)

    file_size = os.stat(file_to_upload).st_size

    if quiet:
        s3_client.upload_file(file_to_upload, bucket, _location, Config=config)
    else:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Uploading', leave=False) as pbar:
            s3_client.upload_file(file_to_upload, bucket, _location,
                                  Config=config, Callback=pbar.update)


@s3_client_function
def rename(bucket, s3_client, source_key, dest_key, verbose=True):
    '''
    Move file matching source_key to dest_key
    '''

    if source_key == dest_key:
        LOGGER.error('Source and destination are the same file!')
        return False

    copy_source = {'Bucket': bucket, 'Key': source_key}
    s3_client.copy(copy_source, bucket, dest_key)
    if file_exists(bucket, s3_client, dest_key):
        if verbose:
            LOGGER.info(f'{source_key} -> {dest_key}')
        s3_client.delete_object(Bucket=bucket, Key=source_key)
        if verbose:
            LOGGER.info(f'Deleted {source_key}')
        return True
    else:
        LOGGER.error('Failed to move {source_key} -> {dest_key}')
        return False


@s3_client_function
def list_files(bucket, s3_client, prefix=None):
    '''
    List file(s) in an s3 bucket.

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    prefix: str
        Only files with the prefix will be returned.

    Returns
    -------
    A generator to a list of files.
    '''

    kwargs = dict()
    if prefix:
        kwargs['Prefix'] = prefix

    response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
    if 'Contents' in response:
        yield response['Contents']
    while response['IsTruncated']:
        kwargs['ContinuationToken'] = response['NextContinuationToken']
        response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
        yield response['Contents']


@s3_client_function
def list_files_depth(bucket, s3_client, max_depth=None, prefix='/'):
    '''
    List file(s) in an s3 bucket to a specific depth.

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    max_depth: int
        Directory depth to list to. Default is 1.
    prefix: str
        Only files with the prefix will be returned.

    Returns
    -------
    A generator to a list of files.
    '''

    def list_directory(prefix, current_depth):
        paginator = s3_client.get_paginator('list_objects_v2')
        for response in paginator.paginate(Bucket=bucket, Delimiter='/', Prefix=prefix):

            # process files
            files = []
            if 'Contents' in response:
                files = response['Contents']

            # process subdirectories
            directories = []
            if 'CommonPrefixes' in response:
                directories = [d['Prefix'] for d in response['CommonPrefixes']]

            if len(files) > 0:
                yield files

            for directory in directories:
                yield [{'Key': directory, 'Size': 0, 'IsDirectory': True}]
                if max_depth is None or current_depth < max_depth:
                    yield from list_directory(f'/{directory.strip("/")}/', current_depth + 1)

    yield from list_directory(prefix, 0)


@s3_client_function
def list_versions(bucket, s3_client, file_name=None):
    '''
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    prefix: str
        Only files with the prefix will be returned.

    Returns
    -------
    A generator to a list of file versions.
    '''

    kwargs = dict()
    if file_name:
        kwargs['Prefix'] = file_name

    response = s3_client.list_object_versions(Bucket=bucket, **kwargs)
    if 'Versions' in response:
        yield response['Versions']
    while response['IsTruncated']:
        kwargs['ContinuationToken'] = response['NextContinuationToken']
        response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
        yield response['Versions']


@s3_client_function
def delete_files(bucket, s3_client, files, files_per_request=1000, verbose=False):
    '''
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    files: list
        A list of file keys to delete.
    files_per_request: int
        Number of files to put in a single http request.
        Default is 1000. Maximum is 1000.
    verbose: bool

    Returns
    -------
    If verbose, yields the response for each delete request.
    If not verbose, None
    '''

    if files_per_request > 1000:
        raise ValueError('Maxium files per request is 1000.')

    # split list of files into chunks
    files = [{'Key':x} for x in files]
    file_chunks = [files[i:i+files_per_request] for i in range(0, len(files), files_per_request)]

    # delete the chunks
    for chunk in file_chunks:
        response = s3_client.delete_objects(Bucket=bucket, Delete={'Objects':chunk})
        if verbose:
            yield response


def get_file(bucket, s3_client, s3_file, local_file):
    try:

        # s3_client.download_file(bucket, s3_file, local_file)

        file_size = s3_client.head_object(Bucket=bucket, Key=s3_file)['ContentLength']

        # Download the file with progress monitoring
        with tqdm(total=file_size, unit='B', unit_scale=True, desc='Downloading', leave=False) as pbar:
            s3_client.download_file(bucket, s3_file, local_file, Callback=pbar.update)

        LOGGER.info(f"\nFile downloaded successfully: {local_file}")
        return True

    except (ClientError, FileNotFoundError) as e:
        LOGGER.error(str(e))
        return False


class Main(object):
    '''
    A class to parse subcommands.
    Inspired by this blog post: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    '''

    LIST_DESCRIPTION = 'List files in bucket or subdirectory.'
    MD5_DESCRIPTION = 'Get or calculate file md5 hash.'
    UPLOAD_DESCRIPTION = 'Upload file(s).'
    DOWNLOAD_DESCRIPTION = 'Download file(s).'
    DELETE_DESCRIPTION = 'Delete file(s).'
    MOVE_DESCRIPTION = 'Move file'

    def __init__(self):
        parser = argparse.ArgumentParser(description='Command line client for AWS s3.',
                                         usage = f'''PDC_client -b <bucket_name> -k <access_key> -s <secret_key> <subcommand> [<options>]

Available commands:
   ls/list      {Main.LIST_DESCRIPTION}
   md5          {Main.MD5_DESCRIPTION}
   put/upload   {Main.UPLOAD_DESCRIPTION}
   get/download {Main.DOWNLOAD_DESCRIPTION}
   rm/delete    {Main.DELETE_DESCRIPTION}
   mv/move      {Main.MOVE_DESCRIPTION}''')
        parser.add_argument('--debug', choices = ['pdb', 'pudb'], default=None,
                            help='Start the main method in selected debugger')
        parser.add_argument('-b', '--bucket', type=str, required=True, help='s3 bucket name.')
        parser.add_argument('-k', '--accessKey', type=str, required=True, help='AWS access key.')
        parser.add_argument('-s', '--secretAccessKey', type=str, required=True, help='AWS secret access key.')
        parser.add_argument('command', help = 'Subcommand to run.')
        subcommand_start = _firstSubcommand(sys.argv)
        args = parser.parse_args(sys.argv[1:(subcommand_start + 1)])

        if args.debug:
            if args.debug == 'pdb':
                import pdb as debugger
            elif args.debug == 'pudb':
                import pudb as debugger
            debugger.set_trace()

        if not args.command in SUBCOMMANDS:
            LOGGER.error(f'ERROR: {args.command} is an unknown command!\n')
            parser.print_help()
            sys.exit(1)

        self.client = client('s3', aws_access_key_id=args.accessKey,
                                   aws_secret_access_key=args.secretAccessKey)
        self.bucket = args.bucket

        getattr(self, args.command)(subcommand_start + 1)


    def move(self, subcommand_start):
        parser = argparse.ArgumentParser()
        parser.add_argument('-q', '--quiet', action='store_true', default=False,
                            help='Less verbose output.')
        parser.add_argument('file')
        parser.add_argument('dest')
        args = parser.parse_args(sys.argv[subcommand_start:])

        if file_exists(self.bucket, self.client, args.file):
            if not rename(self.bucket, self.client, args.file, args.dest, verbose=not args.quiet):
                sys.exit(1)


    def mv(self, subcommand_start):
        self.move(subcommand_start)


    def list(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-l', action='store_true', default=False, help='Use a long listing format.')
        parser.add_argument('--si', action='store_true', default=False,
                            help='Print file sizes in powers of 1000, not 1024.')
        parser.add_argument('-H', action='store_true', default=False,
                            help='With -l, print file sizes in human readable format.')
        parser.add_argument('-t', '--fileType', default='both', choices=['f', 'd', 'fd'],
                            help='File type to list. "fd" is the default.')
        parser.add_argument('-d', '--maxDepth', default=None, type=int,
                            help='Maximum directory depth to list when using --recursive option.')
        parser.add_argument('-r', '--recursive', default=False, action='store_true',
                            help='Recursively list files.')
        # parser.add_argument('--versions', action='store_true', default=False, help='Also print file versions.')
        parser.add_argument('prefix', nargs='*',
                            help='Subdirectory/ies to list. If none, the entire contents of the bucket are listed.')
        args = parser.parse_args(sys.argv[subcommand_start:])

        list_dirs = args.prefix if len(args.prefix) >= 1 else ('/',)

        max_depth = args.maxDepth if args.recursive else 1

        for d in list_dirs:
            if len(list_dirs) > 1:
                if args.fileType != 'f':
                    sys.stdout.write(f'{d}:\n')

            # list_f = list_versions if args.versions else list_files
            for chunk in list_files_depth(self.bucket, self.client, max_depth, d.rstrip('/') + '/'):
                for file in chunk:
                    if 'IsDirectory' in file:
                        time = '\t\t'
                        if args.fileType == 'f':
                            continue
                    else:
                        # convert last modified time to local timezone and format
                        time = datetime.astimezone(file["LastModified"]).strftime("%b %d %Y %H:%m")
                        if args.fileType == 'd':
                            continue

                    if args.l:
                        # convert file size to human readable format if necissary
                        size = format_size(file["Size"], args.si) if args.H else file["Size"]

                        sys.stdout.write(f'{size}\t{time}\t')
                    # if args.versions:
                    #     sys.stdout.write(f'{file["VersionId"]}\t')
                    sys.stdout.write(f'{file["Key"]}\n')


    def ls(self, subcommand_start):
        self.list(subcommand_start)


    def upload(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-q', '--quiet', action='store_true', default=False,
                            help='Less verbose output.')
        parser.add_argument('-f', '--force', action='store_true', default=False,
                            help="Overite file if it already exists.")
        parser.add_argument('--threads', type=int, default=1,
                            help='Number of files to upload in parallel.')
        parser.add_argument('files', nargs='+',
                            help='File(s) to upload')
        parser.add_argument('directory', help='Directory on bucket to upload to.')
        args = parser.parse_args(sys.argv[subcommand_start:])

        # with Pool(processes=min(args.threads, len(args.files)):
        for file in args.files:
            remote_path = f'{args.directory.rstrip("/")}/{os.path.basename(file)}'
            this_file_exists = file_exists(self.bucket, self.client, remote_path)
            if not args.quiet:
                LOGGER.info(f'{"Overwriting" if this_file_exists else "Uploading"}: "{file}"')
            if this_file_exists and not args.force:
                if not args.quiet:
                    LOGGER.info(f'"{remote_path}" already exists on bucket. Skipping...')
                continue

            upload_file(self.bucket, self.client, file, args.directory, quiet=args.quiet)
            if not args.quiet:
                LOGGER.info(f'Finished uploading "{file}"')


    def put(self, subcommand_start):
        self.upload(subcommand_start)


    def delete(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.DELETE_DESCRIPTION)
        parser.add_argument('-q', '--quiet', action='store_true', default=False,
                            help='Less verbose output.')
        parser.add_argument('-f', '--force', action='store_true', default=False,
                            help='Ignore nonexistent files.')
        parser.add_argument('files', nargs='+',
                            help='Remote file(s) to delete')
        args = parser.parse_args(sys.argv[subcommand_start:])

        for response in delete_files(self.bucket, self.client, args.files, verbose=True):
            if not args.quiet:
                for file in response['Deleted']:
                    key = file['Key']
                    LOGGER.info(f'Deleted "{key}"')
            if 'Errors' in response:
                LOGGER.error(f'There were {len(response["Errors"])} errors deleteing files in this chunk!')
                for error in response['Errors']:
                    LOGGER.error(f'{error["Message"]}')
                    if response['Error']['Code'] == 'NoSuchKey' and args.force:
                        continue
                    sys.exit(1)


    def rm(self, subcommand_start):
        self.delete(subcommand_start)


    def md5(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.MD5_DESCRIPTION)
        parser.add_argument('-c', '--calculate', action='store_true', default=False,
                            help='If the ETag is not a md5 hash downloaded the file in pieces '
                                 'to calculate the hash manually.')
        parser.add_argument('-q', '--quiet', action='store_true', default=False,
                            help='Less verbose output.')

        parser.add_argument('files', nargs='+', help='Remote file(s)')
        args = parser.parse_args(sys.argv[subcommand_start:])

        for file in args.files:
            md5_sum = get_s3_file_md5(self.bucket, self.client, file,
                                      calculate=args.calculate, verbose=not args.quiet)
            if md5_sum is None:
                if args.calculate:
                    LOGGER.error(f"An unknown error occured getting the hash for '{file}'")
                else:
                    LOGGER.error(f"'{file}' checksum is not a md5 hash! Use --calculate to calculate it manually.")
                sys.exit(1)

            sys.stdout.write(f'{md5_sum}\t{file}\n')


    def download(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.DELETE_DESCRIPTION)
        parser.add_argument('-q', '--quiet', action='store_true', default=False,
                            help='Less verbose output.')
        parser.add_argument('-f', '--force', action='store_true', default=False,
                            help='Overwrite files at destination.')
        parser.add_argument('file', help='Remote file to download.')
        parser.add_argument('dest', nargs='?', default=None, help='destination')
        args = parser.parse_args(sys.argv[subcommand_start:])

        dest = None
        if args.dest is None:
            dest = os.path.basename(args.file)
        elif not os.path.isdir(args.dest) or not os.path.isdir(os.path.dirname(args.dest)):
            LOGGER.error(f'Target file: "{args.dest}" could not be created!')
            sys.exit(1)
        elif os.path.isdir(args.dest):
            dest = f'{args.dest}/{os.path.basename(args.file)}'
        else:
            dest = args.dest

        get_file(self.bucket, self.client, args.file, dest)


    def get(self, subcommand_start):
        self.download(subcommand_start)


if __name__ == '__main__':
    _ = Main()


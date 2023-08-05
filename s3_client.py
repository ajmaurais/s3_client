
import argparse
import sys
from os.path import basename
from datetime import datetime
import logging
from functools import wraps

from boto3 import client
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s"
)
LOGGER = logging.getLogger()

SUBCOMMANDS = {'ls', 'list', 'put', 'upload', 'get', 'download', 'rm', 'delete'}


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
    file_exists: boo
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


@s3_client_function
def upload_file(bucket, s3_client, file_to_upload, location):
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
    _location = f'{location.rstrip("/")}/{basename(file_to_upload)}'
    GB = 1024 ** 3
    config=TransferConfig(multipart_threshold=5*GB)
    response = s3_client.upload_file(file_to_upload, bucket, _location, Config=config)


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


class Main(object):
    '''
    A class to parse subcommands.
    Inspired by this blog post: https://chase-seibert.github.io/blog/2014/03/21/python-multilevel-argparse.html
    '''

    LIST_DESCRIPTION = 'List files in bucket or subdirectory.'
    UPLOAD_DESCRIPTION = 'Upload file(s).'
    DELETE_DESCRIPTION = 'Delete file(s).'

    def __init__(self):
        parser = argparse.ArgumentParser(description='Command line client for AWS s3.',
                                         usage = f'''PDC_client -b <bucket_name> -k <access_key> -s <secret_key> <subcommand> [<options>]

Available commands:
   ls/list      {Main.LIST_DESCRIPTION}
   put/upload   {Main.UPLOAD_DESCRIPTION}
   rm/delete    {Main.DELETE_DESCRIPTION}''')
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


    def list(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-l', action='store_true', default=False, help='Use a long listing format.')
        parser.add_argument('--si', action='store_true', default=False,
                            help='Print file sizes in powers of 1000, not 1024.')
        parser.add_argument('-H', action='store_true', default=False,
                            help='With -l, print file sizes in human readable format.')
        parser.add_argument('--versions', action='store_true', default=False, help='Also print file versions.')
        parser.add_argument('prefix', nargs='*',
                            help='Subdirectory/ies to list. If none, the entire contents of the bucket are listed.')
        args = parser.parse_args(sys.argv[subcommand_start:])

        list_dirs = args.prefix if len(args.prefix) >= 1 else (None,)
        
        for d in list_dirs:
            if len(list_dirs) > 1:
                sys.stdout.write(f'{d}:\n')

            list_f = list_versions if args.versions else list_files
            for chunk in list_f(self.bucket, self.client, d):
                for file in chunk:
                    if args.l:
                        # convert file size to human readable format if necissary
                        size = format_size(file["Size"], args.si) if args.H else file["Size"]
                        # convert modified time to local timezone and format
                        time = datetime.astimezone(file["LastModified"]).strftime("%b %d %Y %H:%m")
                        sys.stdout.write(f'{size}\t{time}\t')
                    if args.versions:
                        sys.stdout.write(f'{file["VersionId"]}\t')
                    sys.stdout.write(f'{file["Key"]}\n')


    def ls(self, subcommand_start):
        self.list(subcommand_start) 


    def upload(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='Print verbose output.')
        parser.add_argument('-f', '--force', action='store_true', default=False,
                            help="Overite file if it already exists.")
        parser.add_argument('files', nargs='+',
                            help='File(s) to upload')
        parser.add_argument('directory', help='Directory on bucket to upload to.')
        args = parser.parse_args(sys.argv[subcommand_start:])

        for file in args.files:
            if args.verbose:
                LOGGER.info(f'Uploading: "{file}"')
            if not args.force and file_exists(self.bucket, self.client, f'{args.directory.rstrip("/")}/{basename(file)}'):
                LOGGER.info(f'"{file}" already exists on bucket. Skipping...')
                continue

            upload_file(self.bucket, self.client, file, args.directory)
            if args.verbose:
                LOGGER.info(f'Finished uploading "{file}"')


    def put(self, subcommand_start):
        self.upload(subcommand_start)


    def delete(self, subcommand_start):
        parser = argparse.ArgumentParser(description=Main.DELETE_DESCRIPTION)
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='Print verbose output.')
        parser.add_argument('-f', '--force', action='store_true', default=False,
                            help='Ignore nonexistent files.')
        parser.add_argument('files', nargs='+',
                            help='Remote file(s) to delete')
        args = parser.parse_args(sys.argv[subcommand_start:])

        for response in delete_files(self.bucket, self.client, args.files, verbose=True):
            if args.verbose:
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


    def download(self, subcommand_start):
        raise RuntimeError('Sorry, not yet implemented.')


    def get(self, subcommand_start):
        self.download(subcommand_start)


if __name__ == '__main__':
    _ = Main()



import argparse
import sys
import os
from datetime import datetime
import logging

from time import sleep

import boto3
from botocore.exceptions import ClientError
from boto3.s3.transfer import TransferConfig


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s"
)
LOGGER = logging.getLogger()

SUBCOMMANDS = {'ls', 'list', 'upload', 'rm', 'delete'}


def _firstSubcommand(argv):
    for i in range(1, len(argv)):
        if argv[i] in SUBCOMMANDS:
            return i
    return len(argv)


def format_size(size, si=False):
    '''
    Format file size in bytes to human readable format.
    '''
    divisor = 1000 if si else 1024.0
    for i, unit in enumerate(("", "K", "M", "G", "T", "P")):
        if i > 1:
            size /= divisor
        if abs(size) < divisor:
            size_f = round(size, 1)
            if size_f % 1 == 0 or size_f >= 100:
                size_f = round(size)

            return f"{size_f}{unit.lower() if si else unit}"
    return f'{size:.0e}{"p" if si else "P"}'


def upload_file(bucket, s3_client, file_name):
    '''
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    file_name: str
        File to upload
    '''

    # Upload the file
    try:
        GB = 1024 ** 3
        config=TransferConfig(multipart_threshold=5*GB)
        response = s3_client.upload_file(file_name, bucket, file_name, Config=config)

    except ClientError as e:
        LOGGER.error(e)
        sys.exit(1)


def list_files(bucket, s3_client, file_name=None):
    '''
    Upload a file to an S3 bucket

    Parameters
    ----------
    bucket: str
        Name of bucket to upload to
    s3_client: boto3.client
        Initialized client object
    file_name: str
        File to upload. If none, the entire 
    '''
    
    kwargs = dict()
    files = list()
    if file_name:
        kwargs['Prefix'] = file_name

    try:
        response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
        if 'Contents' in response:
            files = response['Contents']
            yield files
        while response['IsTruncated']:
            kwargs['Marker'] = response['Marker']
            response = s3_client.list_objects_v2(Bucket=bucket, **kwargs)
            yield response['Contents']

    except ClientError as e:
        LOGGER.error(e)
        sys.exit(1)


def delete_file(file_name, bucket, s3_client):
    pass


def _firstSubcommand(argv):
    for i in range(1, len(argv)):
        if argv[i] in SUBCOMMANDS:
            return i
    return len(argv)


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
   list     {Main.LIST_DESCRIPTION}
   upload   {Main.UPLOAD_DESCRIPTION}
   delete   {Main.DELETE_DESCRIPTION}''')
        parser.add_argument('--debug', choices = ['pdb', 'pudb'], default=None,
                            help='Start the main method in selected debugger')
        parser.add_argument('-b', '--bucket', required=True, help='s3 bucket name.')
        parser.add_argument('-k', '--accessKey', required=True, help='AWS access key.')
        parser.add_argument('-s', '--secretAccessKey', required=True, help='AWS secret access key.')
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

        self.client = boto3.client('s3', aws_access_key_id=args.accessKey,
                                   aws_secret_access_key=args.secretAccessKey)
        self.bucket = args.bucket

        getattr(self, args.command)(subcommand_start + 1)


    def list(self, start=2):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-l', action='store_true', default=False, help='Use a long listing format.')
        parser.add_argument('--si', action='store_true', default=False,
                            help='Print file sizes in powers of 1000, not 1024.')
        parser.add_argument('-H', action='store_true', default=False,
                            help='With -l, print file sizes in human readable format.')
        parser.add_argument('subdirectory', nargs='*',
                            help='Subdirectory/ies to list. If none, the entire contents of the bucket are listed.')
        args = parser.parse_args(sys.argv[start:])

        for chunk in list_files(self.bucket, self.client):
            for file in chunk:
                if args.l:
                    # convert file size to human readable format if necissary
                    size = format_size(file["Size"], args.si) if args.H else file["Size"]
                    # convert modified time to local timezone and format
                    time = datetime.astimezone(file["LastModified"]).strftime("%b %d %Y %H:%m")
                    sys.stdout.write(f'{size}\t{time}\t')
                sys.stdout.write(f'{file["Key"]}\n')


    def ls(self, start=2):
        self.list(start) 


    def upload(self, start=2):
        parser = argparse.ArgumentParser(description=Main.LIST_DESCRIPTION)
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                            help='Print verbose output.')
        parser.add_argument('files', nargs='+',
                            help='File(s) to upload')
        args = parser.parse_args(sys.argv[start:])

        for file in args.files:
            if args.verbose:
                LOGGER.info(f'Uploading: "{file}"')
            upload_file(self.bucket, self.client, file)
            if args.verbose:
                LOGGER.info(f'Finished uploading "{file}"')


if __name__ == '__main__':
    _ = Main()


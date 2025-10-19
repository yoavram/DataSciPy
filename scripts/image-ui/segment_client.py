import io

import requests
import click

base_url = "http://127.0.0.1:5000/api/1"

def error(msg):
    """Display error message
    """
    click.secho(msg, fg="red")

def upload_image(filename):
    """Upload image to web server, returning its URL.
    
    Parameters
    ----------
    filename : str
        filename of image file to upload
        
    Returns
    -------
    str
        URL of the image on the remote server
        
    Raises
    ------
    Exception :
        Exception with message in case upload failed.
    IOError : 
        In case there was a problem opening the file.
    """
    url = "{}/image".format(base_url)
    with open(filename, 'rb') as f:
        resp = requests.post(url, files={'file': f})        
    if not resp.ok:
        raise Exception(resp.reason)
    return resp.json()['url']

def download_image(url, filename=None):
    """Download image from web server.
    
    Parameters
    ----------
    url : str
        URL of image to download
    filename : str
        filename to write the downloaded image to, defaults to `None`
        
    Returns
    -------
    io.BytesIO 
        the image data if `filename` is `None`, otherwise `None`
        
    Raises
    ------
    Exception :
        Exception with message in case download failed.
    IOError : 
        In case there was a problem opening the file.
    """
    resp = requests.get(url)
    if not resp.ok:
        raise Exception(resp.reason)
    if filename is not None:
        with open(filename, "wb") as f:
            f.write(resp.content)
    else:
        return io.BytesIO(resp.content)
        
def segment_image(url):
    """Ask web server to segment an image, returning the URL of the segmented image.
    
    Parameters
    ----------
    url : str
        URL of the image to segment on the remote server
        
    Returns
    -------
    str
        URL of the segmented image on the remote server
        
    Raises
    ------
    Exception :
        Exception with message in case segmentation failed.
    """
    url = "{}/segment".format(url)
    resp = requests.get(url)
    if not resp.ok:
        raise Exception(resp.reason)
    return resp.json()['url']

@click.command()
@click.argument("src", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument("dst", type=click.Path(writable=True, dir_okay=False))
def main(src, dst):    
    url = upload_image(src)
    url = segment_image(url)
    download_image(url, dst)

if __name__ == '__main__':
    main()
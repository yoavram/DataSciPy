import io

import requests
import click

base_url = "http://127.0.0.1:5000/api/2"

def error(msg):
    click.secho(msg, fg="red")

def get_token(password):
    resp = requests.get("{}/token".format(base_url), auth=('', password))
    if not resp.ok:
        raise click.UsageError("Failed getting token")
    return resp.json()['token']  
    
def upload_image(filename, token):
    url = "{}/image".format(base_url)
    with open(filename, 'rb') as f:
        resp = requests.post(url, files={'file': f}, auth=(token, ''))        
    if not resp.ok:
        raise click.UsageError(resp.reason)
    return resp.json()['image_id']

def download_image(url, token, filename=None):
    resp = requests.get(url, auth=(token, ''))
    if not resp.ok:
        raise click.UsageError(resp.reason)
    if filename is not None:
        with open(filename, "wb") as f:
            f.write(resp.content)
    else:
        return io.BytesIO(resp.content)
        
def segment_image(image_id, token):
    url = "{}/image/{}/segment".format(base_url, image_id)
    resp = requests.get(url, auth=(token, ''))
    if not resp.ok:
        raise click.UsageError(resp.reason)
    return resp.json()['url']

@click.command()
@click.argument("src", type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument("dst", type=click.Path(writable=True, dir_okay=False))
@click.option('--password', prompt=True, hide_input=True)
def main(src, dst, password):
    token = get_token(password)        
    image_id = upload_image(src, token)
    url = segment_image(image_id, token) 
    download_image(url, token, dst)

if __name__ == '__main__':
    main()
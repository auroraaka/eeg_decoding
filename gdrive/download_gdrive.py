code = '421107498268-i50dh342lvl9ur9pv8sqq2ml2trb7u8f.apps.googleusercontent.com'
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import os

def authenticate():
    gauth = GoogleAuth()
    gauth.settings['client_config_file'] = 'client_secret_421107498268-i50dh342lvl9ur9pv8sqq2ml2trb7u8f.apps.googleusercontent.com.json'
    gauth.LocalWebserverAuth()
    return GoogleDrive(gauth)

def download_folder(drive, folder_id, local_path):
    os.makedirs(local_path, exist_ok=True)
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    for f in file_list:
        if f['mimeType']=='application/vnd.google-apps.folder':
            download_folder(drive, f['id'], os.path.join(local_path, f['title']))
        else:
            print(f'Downloading {f["title"]} to {local_path}')
            f.GetContentFile(os.path.join(local_path, f['title']))

if __name__ == "__main__":
    drive = authenticate()
    folder_id = '12SNgscNFPkg194_vcD3uv-0fJcUExhmv'
    local_download_path = 'checkpoint'
    os.makedirs(local_download_path, exist_ok=True)
    download_folder(drive, folder_id, local_download_path)


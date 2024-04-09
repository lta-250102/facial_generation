import subprocess
import gdown
import os

subprocess.call(['apt', 'update'])
subprocess.call(['apt', 'install', 'unzip'])
subprocess.call(['pip', 'install', '-r', 'requirements.txt'])

# data
os.makedirs('./data/celebA/', exist_ok=True)
url = 'https://drive.google.com/file/d/1BguZoSwMbNjvndJZq14_3JjPSv7orNcV/view?usp=sharing'
output = './data/celebA/img_align_celeba.zip'
gdown.download(url, output, quiet=False)

subprocess.call(['unzip', 'data/celebA/img_align_celeba.zip', '-d', 'data/celebA/'])

url = 'https://drive.google.com/file/d/12qt6jjehowlD5USiQ_Hveuco2KSdq5YJ/view?usp=sharing'
output = './data/celebA/attrs.json'
gdown.download(url, output, quiet=False)

url = 'https://drive.google.com/file/d/13SZwAOOOceFzg9laR8G7QYXowsoeB9im/view?usp=sharing'
output = './data/celebA/captions.json'
gdown.download(url, output, quiet=False)

# state dict
url = 'https://drive.google.com/file/d/1dd6rwByW4w9bkH6b2BjGRWtmh9SI1NSN/view?usp=sharing'
output = './pretrained/colla_module.pt'
gdown.download(url, output, quiet=False)


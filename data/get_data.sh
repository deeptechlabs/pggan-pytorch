#!/bin/bash

# Ensure that 7z is installed
# Ensure that you have the google_drive.py script

# CelebA
mkdir celeba
python google_drive.py 0B7EVK8r0v71pckZsdFFIYlJoN1k celeba/img_align_celeba_png.7z.016
python google_drive.py 0B7EVK8r0v71pczZ0NFNFdFRXSUU celeba/img_align_celeba_png.7z.015
python google_drive.py 0B7EVK8r0v71pbFlZaURkY3dhWWM celeba/img_align_celeba_png.7z.014
python google_drive.py 0B7EVK8r0v71pNjFfTGYzTWJDdUU celeba/img_align_celeba_png.7z.013
python google_drive.py 0B7EVK8r0v71pSk1zVWN2aHhMZ3c celeba/img_align_celeba_png.7z.012
python google_drive.py 0B7EVK8r0v71pZWZ4UGdBbk9UVWs celeba/img_align_celeba_png.7z.011
python google_drive.py 0B7EVK8r0v71pLThPNzFETUNMUVE celeba/img_align_celeba_png.7z.010
python google_drive.py 0B7EVK8r0v71pZk5QcUlObVltaEE celeba/img_align_celeba_png.7z.009
python google_drive.py 0B7EVK8r0v71pejhuem9QV2h0MDQ celeba/img_align_celeba_png.7z.008
python google_drive.py 0B7EVK8r0v71pNE5aQmY5c2ZLOXc celeba/img_align_celeba_png.7z.007
python google_drive.py 0B7EVK8r0v71pa3NIcEgtTXZrM0U celeba/img_align_celeba_png.7z.006
python google_drive.py 0B7EVK8r0v71pVjRlNVB3cDVjaDQ celeba/img_align_celeba_png.7z.005
python google_drive.py 0B7EVK8r0v71pVmYwbmRtV2hZcDA celeba/img_align_celeba_png.7z.004
python google_drive.py 0B7EVK8r0v71peUlHSDVhd0JTamM celeba/img_align_celeba_png.7z.003
python google_drive.py 0B7EVK8r0v71pR2NwRnU2cVZ2RTg celeba/img_align_celeba_png.7z.002
python google_drive.py 0B7EVK8r0v71pSVd0ZjQ3Sks2dzg celeba/img_align_celeba_png.7z.001
7z x celeba/img_align_celeba_png.7z.001 -oceleba/
rm celeba/img_align_celeba_png.7z.001
rm celeba/img_align_celeba_png.7z.002
rm celeba/img_align_celeba_png.7z.003
rm celeba/img_align_celeba_png.7z.004
rm celeba/img_align_celeba_png.7z.005
rm celeba/img_align_celeba_png.7z.006
rm celeba/img_align_celeba_png.7z.007
rm celeba/img_align_celeba_png.7z.008
rm celeba/img_align_celeba_png.7z.009
rm celeba/img_align_celeba_png.7z.010
rm celeba/img_align_celeba_png.7z.011
rm celeba/img_align_celeba_png.7z.012
rm celeba/img_align_celeba_png.7z.013
rm celeba/img_align_celeba_png.7z.014
rm celeba/img_align_celeba_png.7z.015
rm celeba/img_align_celeba_png.7z.016

python google_drive.py 0B7EVK8r0v71pcTFwT1VFZzkzZk0 celeba/img_celeba.7z.014
python google_drive.py 0B7EVK8r0v71paXBad2lfSzlzSlk celeba/img_celeba.7z.013
python google_drive.py 0B7EVK8r0v71pTVd3R2NpQ0FHaGM celeba/img_celeba.7z.012
python google_drive.py 0B7EVK8r0v71pNU9BZVBEMF9KN28 celeba/img_celeba.7z.011
python google_drive.py 0B7EVK8r0v71pa09rcFF4THRmSFU celeba/img_celeba.7z.010
python google_drive.py 0B7EVK8r0v71pVlg5SmtLa1ZiU0k celeba/img_celeba.7z.009
python google_drive.py 0B7EVK8r0v71pLVltX2F6dzVwT0E celeba/img_celeba.7z.008
python google_drive.py 0B7EVK8r0v71pQlZrOENSOUhkQ3c celeba/img_celeba.7z.007
python google_drive.py 0B7EVK8r0v71pbWFfbGRDOVZxOUU celeba/img_celeba.7z.006
python google_drive.py 0B7EVK8r0v71pMktaV1hjZUJhLWM celeba/img_celeba.7z.005
python google_drive.py 0B7EVK8r0v71peXc4WldxZGFUbk0 celeba/img_celeba.7z.004
python google_drive.py 0B7EVK8r0v71pMk5FeXRlOXcxVVU celeba/img_celeba.7z.003
python google_drive.py 0B7EVK8r0v71peFphOHpxODd5SjQ celeba/img_celeba.7z.002
python google_drive.py 0B7EVK8r0v71pQy1YUGtHeUM2dUE celeba/img_celeba.7z.001
7z x celeba/img_celeba.7z.001 -oceleba/
rm celeba/img_celeba.7z.014
rm celeba/img_celeba.7z.013
rm celeba/img_celeba.7z.012
rm celeba/img_celeba.7z.011
rm celeba/img_celeba.7z.010
rm celeba/img_celeba.7z.009
rm celeba/img_celeba.7z.008
rm celeba/img_celeba.7z.007
rm celeba/img_celeba.7z.006
rm celeba/img_celeba.7z.005
rm celeba/img_celeba.7z.004
rm celeba/img_celeba.7z.003
rm celeba/img_celeba.7z.002
rm celeba/img_celeba.7z.001

# CelebA-HQ deltas
mkdir celeba-hq-deltas
python google_drive.py 0B4qLcYyJmiz0U2hZTEJfaEZSMFE celeba-hq-deltas/README.txt
python google_drive.py 0B4qLcYyJmiz0U25vdEVIU3NvNFk celeba-hq-deltas/image_list.txt
python google_drive.py 0B4qLcYyJmiz0U1pnMEI4WXN1S3M celeba-hq-deltas/deltas29000.zip
python google_drive.py 0B4qLcYyJmiz0U1BhWlFGRXc1aHc celeba-hq-deltas/deltas28000.zip
python google_drive.py 0B4qLcYyJmiz0U1BRYl9tSWFWVGM celeba-hq-deltas/deltas27000.zip
python google_drive.py 0B4qLcYyJmiz0U0tBanQ4cHNBUWc celeba-hq-deltas/deltas26000.zip
python google_drive.py 0B4qLcYyJmiz0U0lYX1J1Tk5vMjQ celeba-hq-deltas/deltas25000.zip
python google_drive.py 0B4qLcYyJmiz0U0YwQmluMmJuX2M celeba-hq-deltas/deltas24000.zip
python google_drive.py 0B4qLcYyJmiz0U05ZNG14X3ZjYW8 celeba-hq-deltas/deltas23000.zip
python google_drive.py 0B4qLcYyJmiz0TzZySG9IWlZaeGc celeba-hq-deltas/deltas22000.zip
python google_drive.py 0B4qLcYyJmiz0TzBBNE8xbFhaSlU celeba-hq-deltas/deltas21000.zip
python google_drive.py 0B4qLcYyJmiz0TnJQSS1vZS1JYUE celeba-hq-deltas/deltas20000.zip
python google_drive.py 0B4qLcYyJmiz0TnBtdW83OXRfdG8 celeba-hq-deltas/deltas19000.zip
python google_drive.py 0B4qLcYyJmiz0TmhIUGlVeE5pWjg celeba-hq-deltas/deltas18000.zip
python google_drive.py 0B4qLcYyJmiz0TmZqZXN3UWFkUm8 celeba-hq-deltas/deltas17000.zip
python google_drive.py 0B4qLcYyJmiz0TmVkVGJmWEtVbFk celeba-hq-deltas/deltas16000.zip
python google_drive.py 0B4qLcYyJmiz0TmRZTmZyenViSjg celeba-hq-deltas/deltas15000.zip
python google_drive.py 0B4qLcYyJmiz0Tm5MSUp3ZTZ0aTg celeba-hq-deltas/deltas14000.zip
python google_drive.py 0B4qLcYyJmiz0TlpJU1pleF9zbnM celeba-hq-deltas/deltas13000.zip
python google_drive.py 0B4qLcYyJmiz0Tlhvdl9zYlV4UUE celeba-hq-deltas/deltas12000.zip
python google_drive.py 0B4qLcYyJmiz0TlNyLUtOTzk3QjQ celeba-hq-deltas/deltas11000.zip
python google_drive.py 0B4qLcYyJmiz0TlBCNFU3QkctNkk celeba-hq-deltas/deltas10000.zip
python google_drive.py 0B4qLcYyJmiz0Tl9wNEU2WWRqcE0 celeba-hq-deltas/deltas09000.zip
python google_drive.py 0B4qLcYyJmiz0Tksyd21vRmVqamc celeba-hq-deltas/deltas08000.zip
python google_drive.py 0B4qLcYyJmiz0TjdaV2ZsQU94MnM celeba-hq-deltas/deltas07000.zip
python google_drive.py 0B4qLcYyJmiz0TjVkYkF4dTJRNUk celeba-hq-deltas/deltas06000.zip
python google_drive.py 0B4qLcYyJmiz0TjRzeWlhLVJIYk0 celeba-hq-deltas/deltas05000.zip
python google_drive.py 0B4qLcYyJmiz0TjRxVkZ1NGxHTXc celeba-hq-deltas/deltas04000.zip
python google_drive.py 0B4qLcYyJmiz0TjRWUXVvM3hZZE0 celeba-hq-deltas/deltas03000.zip
python google_drive.py 0B4qLcYyJmiz0TjNRV2dUamd0bEU celeba-hq-deltas/deltas02000.zip
python google_drive.py 0B4qLcYyJmiz0TjAwOTRBVmRKRzQ celeba-hq-deltas/deltas01000.zip
python google_drive.py 0B4qLcYyJmiz0TXdaTExNcW03ejA celeba-hq-deltas/deltas00000.zip

# LSUN
wget https://raw.githubusercontent.com/fyu/lsun/master/download.py
mkdir lsun
mkdir lsun/bedroom
python download.py -c bedroom
unzip bedroom_train_lmdb.zip
unzip bedroom_val_lmdb.zip
mv bedroom_train_lmdb lsun/bedroom
mv bedroom_val_lmdb lsun/bedroom
rm bedroom_train_lmdb.zip
rm bedroom_val_lmdb.zip

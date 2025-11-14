#!/usr/bin/env python3
import os,zipfile,urllib.request,argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def check_exists(url):
 try:return url if (r:=urllib.request.urlopen(url)).status==200 else None
 except:return None

def download(url):
 fname=os.path.join("data/tmp",os.path.basename(url))
 if os.path.exists(fname):print(f"skipping {os.path.basename(url)} (already exists)");return fname
 with urllib.request.urlopen(url) as r,open(fname,"wb") as f,tqdm(total=int(r.getheader("Content-Length",0)),unit="B",unit_scale=True,desc=os.path.basename(url)) as pbar:
  while chunk:=r.read(8192):f.write(chunk);pbar.update(len(chunk))
 return fname

def main(year):
 os.makedirs("data/tmp",exist_ok=True)
 base="https://database.nikonoel.fr/lichess_elite_{:04d}-{:02d}.zip"
 urls=[base.format(year,m) for m in range(1,13)]
 print(f"checking available archives for {year}...")
 with ThreadPoolExecutor(max_workers=8) as ex:urls=[u for u in ex.map(check_exists,urls) if u]
 if not urls:print("no archives found for that year.");return
 print(f"{len(urls)} files available.")
 with ThreadPoolExecutor(max_workers=4) as ex:zips=list(ex.map(download,urls))
 print("extracting and merging...")
 out_path=f"data/elite_games_{year}.pgn"
 with open(out_path,"wb") as out_f:
  for z in zips:
   try:
    with zipfile.ZipFile(z) as zf:
     for name in zf.namelist():
      with zf.open(name) as f:
       for chunk in iter(lambda:f.read(8192),b""):out_f.write(chunk)
   except zipfile.BadZipFile:print(f"warning: skipping corrupted {os.path.basename(z)}")
 print("done:",out_path)

if __name__=="__main__":
 parser=argparse.ArgumentParser()
 parser.add_argument("--year",type=int,required=True,help="year to download (e.g. 2020)")
 args=parser.parse_args()
 main(args.year)

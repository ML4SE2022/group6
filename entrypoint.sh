echo "Downloading JavaCorpus"
cd /dataset/javaCorpus && ./download.sh && python preprocess.py

echo "Downloading PY150"
cd /dataset/py150 && ./download_and_extract.sh &&  python preprocess.py --base_dir=py150_files --output_dir=token_completion

echo "Downloading javascript"
cd /dataset/javascriptAxolotl && python download.py --download true --preprocess true

echo "Downloading typescript"
cd /dataset/typescriptAxolotl && python download.py --download --preprocess

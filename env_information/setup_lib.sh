echo "========== Installing necessary packages =========="
apt-get update
apt-get install nano
apt-get install ffmpeg libsm6 libxext6 -y
apt-get install libsparsehash-dev

echo "============== Installing Anaconda3 ==============="
chmod +x Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh

CONDA_PATH=bash_setup.txt
BASHRC_PATH=~/.bashrc

cat "$CONDA_PATH" >> "$BASHRC_PATH"

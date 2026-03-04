#!/bin/sh
DIR="$(dirname "$(realpath "$0")")"
echo "Downloading models..."
python "$DIR/download_models.py"

cd "$DIR/pam-dual-grosshack"
echo "Building PAM modules..."
rm -rf build
make
sudo make install
cd "$DIR"
chmod +x compile.sh
./compile.sh
echo "Moving system files..."
sudo bash <<EOF
mkdir -p /usr/share/face_recognition

cp -r "$DIR/etc" /

cp "$DIR/facerec.py" "$DIR/daemon.py" /usr/share/face_recognition

cp -r "$DIR/models" /usr/share/face_recognition

chmod +x /usr/share/face_recognition/*.py
echo Restarting services...
systemctl daemon-reload
sudo systemctl reload dbus
systemctl enable org.FaceRecognition
systemctl stop org.FaceRecognition
systemctl start org.FaceRecognition
EOF

echo Done. Anything should be setup if there were no errors.

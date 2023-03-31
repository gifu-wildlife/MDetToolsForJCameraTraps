curl -sc cookie "https://drive.google.com/uc?export=download&id=1s-OfUCNTO3lggtvLDOg7e3Ry5puyZ0Pe" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' cookie)"
curl -Lb cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1s-OfUCNTO3lggtvLDOg7e3Ry5puyZ0Pe" -o sample_data.tar.gz
tar -zxvf sample_data.tar.gz
rm cookie sample_data.tar.gz

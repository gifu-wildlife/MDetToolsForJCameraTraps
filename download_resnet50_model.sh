wget "https://drive.google.com/uc?export=download&id=1deKYwmQz4oUlb3wVw-wbmZDzBbdMTZ79" -O models/classifire/category.txt
curl -sc cookie "https://drive.google.com/uc?export=download&id=1YlNDUAQ4xpUIZ0BV4HtOqAnxEiFCy7iZ" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' cookie)"
curl -Lb cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=1YlNDUAQ4xpUIZ0BV4HtOqAnxEiFCy7iZ" -o models/classifire/15cat_50epoch_resnet50.pth
rm cookie
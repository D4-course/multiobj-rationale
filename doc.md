Running code 
- `sudo docker build -t multi`
- `sudo docker run -v "$(pwd):/app" -it -h multi -p "8000:8000" multi`
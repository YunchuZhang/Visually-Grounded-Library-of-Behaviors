rsync -vaP -e "ssh" *.[p,s][y,h] gauravp@matrix.ml.cmu.edu:~/pytorch_disco
rsync -vaP -e "ssh" nets/*.[p,s][y,h] gauravp@matrix.ml.cmu.edu:~/pytorch_disco/nets/
rsync -vaP -e "ssh" archs/*.[p,s][y,h] gauravp@matrix.ml.cmu.edu:~/pytorch_disco/archs/
rsync -vaP -e "ssh" backend/*.[p,s][y,h] gauravp@matrix.ml.cmu.edu:~/pytorch_disco/backend/

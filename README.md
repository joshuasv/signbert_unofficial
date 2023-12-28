# Download MANO files

1. Go to the [MANO website](http://mano.is.tue.mpg.de/)
2. Create account
3. Download Models & Code
4. Extract the `*.zip` inside the `signbert/model/thirdparty/mano_assets`
5. Folder structure should look like this:
```bash
mano_assets/
    ├── info.txt
    ├── __init__.py
    ├── LICENSE.txt
    ├── models
    │   ├── info.txt
    │   ├── LICENSE.txt
    │   ├── MANO_LEFT.pkl
    │   ├── MANO_RIGHT.pkl
    │   ├── SMPLH_female.pkl
    │   └── SMPLH_male.pkl
    └── webuser
        └── ...
```

# Create virtual environment

```bash
conda env create -f environment.yml
```

_Note, this environment was tested on `Baiona`._

# Run a training session
```bash
python train.py
```

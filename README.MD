# Instalacja

```commandline
pip install -r requirements.txt
vot initialize tests/basic --workspace ./raw/tests/basic
vot initialize vot2022/shorttermbox --workspace ./raw/tests/vot2022stb
vot initialize vot2022/longterm --workspace ./raw/tests/vot2022lt
streamlit run home.py
```

Po instalacji zależności i zbiorów wystarczy:
```commandline
streamlit run home.py
```

## GOTURN

Aby użyć modelu GOTURN należy sciągnąć pliki z repozytorium:
- https://github.com/spmallick/goturn-files

Następnie należy pliki goturn.caffemodel oraz goturn.prototext umieścić w głównym folderze projektu.

## DaSiamRPN

Aby użyć modelu należy ściągnąć pliki:
- https://www.dropbox.com/s/rr1lk9355vzolqv/dasiamrpn_model.onnx?dl=0
- https://www.dropbox.com/s/999cqx5zrfi7w4p/dasiamrpn_kernel_r1.onnx?dl=0
- https://www.dropbox.com/s/qvmtszx5h339a0w/dasiamrpn_kernel_cls1.onnx?dl=0

a następnie umieścić w główym katalogu.

## NanoTrack

Aby użyć modelu należy ściągnąć pliki:
- https://github.com/HonglinChu/SiamTrackers/raw/refs/heads/master/NanoTrack/models/nanotrackv2/nanotrack_backbone_sim.onnx
- https://github.com/HonglinChu/SiamTrackers/raw/refs/heads/master/NanoTrack/models/nanotrackv2/nanotrack_head_sim.onnx

następnie zmień nazwy na:
- backbone.onnx
- neckhead.onnx

oraz umieścić w główym katalogu.

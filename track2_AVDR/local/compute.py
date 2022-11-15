import os 
import logging
from pathlib import Path
from typing import Dict
from typing import List
from typing import Union
import warnings
import subprocess


def read_2column_text(path: Union[Path, str]) -> Dict[str, str]:
    """Read a text file having 2 column as dict object.

    Examples:
        wav.scp:
            key1 /some/path/a.wav
            key2 /some/path/b.wav

        >>> read_2column_text('wav.scp')
        {'key1': '/some/path/a.wav', 'key2': '/some/path/b.wav'}

    """


    data = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for linenum, line in enumerate(f, 1):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 1:
                k, v = sps[0], ""
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f"{k} is duplicated ({path}:{linenum})")
            data[k] = v
    return data

class DatadirWriter:
    """Writer class to create kaldi like data directory.

    Examples:
        >>> with DatadirWriter("output") as writer:
        ...     # output/sub.txt is created here
        ...     subwriter = writer["sub.txt"]
        ...     # Write "uttidA some/where/a.wav"
        ...     subwriter["uttidA"] = "some/where/a.wav"
        ...     subwriter["uttidB"] = "some/where/b.wav"

    """

    def __init__(self, p: Union[Path, str]):
 
        self.path = Path(p)
        self.chilidren = {}
        self.fd = None
        self.has_children = False
        self.keys = set()

    def __enter__(self):
        return self

    def __getitem__(self, key: str) -> "DatadirWriter":

        if self.fd is not None:
            raise RuntimeError("This writer points out a file")

        if key not in self.chilidren:
            w = DatadirWriter((self.path / key))
            self.chilidren[key] = w
            self.has_children = True

        retval = self.chilidren[key]
        return retval

    def __setitem__(self, key: str, value: str):
      
        if self.has_children:
            raise RuntimeError("This writer points out a directory")
        if key in self.keys:
            warnings.warn(f"Duplicated: {key}")

        if self.fd is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.fd = self.path.open("w", encoding="utf-8")

        self.keys.add(key)
        self.fd.write(f"{key} {value}\n")

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.has_children:
            prev_child = None
            for child in self.chilidren.values():
                child.close()
                prev_child = child

        elif self.fd is not None:
            self.fd.close()

configs = dict(
quite =  ["C07","C08","C012"],
onlytv =  ["C00","C01","C02","C09"],
onlytalk = ["C04","C03","C10"],
tv_talk =  ["C05","C06","C11"]
)


def sigle_result(refpath,hypath,tmpdir):
    split_dic= dict(quite={},onlytv={},onlytalk={},tv_talk={})
    hypdic = read_2column_text(hypath)
    for k,v in hypdic.items():
        for discrib,confs in configs.items():
            if k.split("_")[3] in confs:
                split_dic[discrib][k] = v

    with DatadirWriter(tmpdir) as writer:
        for discrib in configs:
            subwriter = writer[discrib+".txt"]
            for k,v in split_dic[discrib].items():
                subwriter[k] = v
    for discrib in configs:
        print(discrib)
        splitfile = discrib+".txt"
        scheduler_order = f"./cer.sh --ref {tmpdir}/{splitfile} --hyp {refpath}"
        return_info = subprocess.Popen(scheduler_order, shell=True, stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        for next_line in return_info.stdout:
            return_line = next_line.decode("utf-8", "ignore")
            if "WER" in return_line:
                print(return_line[5:-11])


tmpdir = "ztest"
rootdir = Path("/raw7/cv1/hangchen2/misp2021_avsr")
for sub_dir in rootdir.rglob("exp/*/predict_*/result_*/decode/scoring_kaldi"):
    print("/".join(sub_dir.parts[6:9]))
    refpath = sub_dir / "test_filt.chars.txt"
    indexpath = sub_dir / "best_cer"
    index = indexpath.read_text().split(" ")[-1].split("_")[-2]
    hypath = sub_dir / f"penalty_0.0/{index}.chars.txt"
    sigle_result(refpath,hypath,tmpdir)
   
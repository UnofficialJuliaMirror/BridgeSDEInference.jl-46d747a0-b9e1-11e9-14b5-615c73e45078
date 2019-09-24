#EDF file: European Data Format
#Data downloaded from https://archive.physionet.org/pn6/chbmit/
using EDF
INPUT = joinpath(Base.source_dir(), "..", "data")
file_name = joinpath(INPUT, "chb01_01.edf")
data =  EDF.read(file_name)


data.header
data.signals

# EDF file: European Data Format
# Data downloaded from https://archive.physionet.org/pn6/chbmit/
# Each file contains 1 hour of data with a frequency observation of 1/250 seconds
# • in between two files there are at most 10 seconds gaps
# • in total there are several hours of data (massive dataframe)
# • data are int16
# • there are several channel, right now we choose one and we model that one

using EDF
INPUT = joinpath(Base.source_dir(), "..", "data")
file_name = joinpath(INPUT, "chb01_01.edf")
data =  EDF.read(file_name)


data.header
data.signals

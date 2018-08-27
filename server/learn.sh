#!/bin/bash

sox -d $1.wav rate 16k silence 1 0.1 3% 1 3.0 3%
mkdir -p ../data/train/$1
mv $1.wav ../data/train/$1

ffmpeg -i ../data/train/$1/$1.wav -f segment -segment_time 5 -c copy ../data/train/$1/out%03d.wav
rm ../data/train/$1
rm ../data/train/$1/$1.wav
for file in ../data/train/$1/*.wav
do
    outfile=${file%.*}
	sox "$file" ${outfile}.r.wav remix 2
	sox "$file" ${outfile}.l.wav remix 1
    sox -m ${outfile}.l.wav ${outfile}.r.wav ${outfile}.mono.wav
	sox ${outfile}.mono.wav -n spectrogram -r -o ${outfile}.mono.png
	rm ${outfile}.l.wav ${outfile}.r.wav ${outfile}.mono.wav
done

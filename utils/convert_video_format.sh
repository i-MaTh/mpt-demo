avconv -y -i /path/to/video.dav -vcodec libx264 -crf 24 -filter:v "setpts=1*PTS" /path/to/video.avi

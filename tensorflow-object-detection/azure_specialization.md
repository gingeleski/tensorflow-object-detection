Making custom Azure VM (more specialized than Deep Learning + Windows based??) for object detection training
------------------------------------------------------------------------------------------------------------
1. Label your stuff with Microsoft VoTT
2. Output it wherever
3. There's a desktop icon that opens this manager
4. With a file select, choose that output folder and any others to train on
5. If needed the labelmap.pbtxt file (or whatever it's called) will be generated automatically
6. One Docker container can generate all the .tfrecord files I would think
7. TODO
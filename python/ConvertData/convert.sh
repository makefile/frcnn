python ./python/ConvertData/LevelDB.py --Type Train 
python ./python/ConvertData/LevelDB.py --Type Test 
python ./python/ConvertData/LMDB.py --Type Train
python ./python/ConvertData/LMDB.py --Type Test
python ./python/ConvertData/PadCifar10.py --Type Train --Pad 4
python ./python/ConvertData/PadCifar10.py --Type Test --Pad 4
python ./python/ConvertData/PadBinaryMean.py

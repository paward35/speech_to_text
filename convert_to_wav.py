import os
from multiprocessing import Pool
from pydub import AudioSegment
from glob import glob
import io

PATH = os.getcwd()
flac_files = [y for x in os.walk(PATH) for y in glob(os.path.join(x[0], '*.flac'))]

def worker(filename):
    flac = AudioSegment.from_file(filename, format='flac')
    stream = io.BytesIO()
    flac.export(stream, format='wav')
    # Save the converted file, assuming you want to replace the extension
    new_filename = filename.replace(".flac", ".wav")
    with open(new_filename, "wb") as f_out:
        f_out.write(stream.getvalue())


if __name__ == "__main__":
    if flac_files:
        with Pool(processes=os.cpu_count()) as p:
            p.map(worker, flac_files)


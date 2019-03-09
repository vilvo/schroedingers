import urllib.request

SOI = b'\xff\xd8'  # start of image in mjpeg byte stream
EOI = b'\xff\xd9'  # end of image in mjpeg byte stream


def client(url):
    """
    simple MJPEG stream client
    Reads the bytestream for SOI and EOI markers and yields frames to generator
    :param url: URL of the HTTP MJPEG stream
    """
    stream = urllib.request.urlopen(url)
    chunk = bytes()
    while True:
        chunk += stream.read(1024)
        a = chunk.find(SOI)
        b = chunk.find(EOI)
        if a != -1 and b != -1:
            image = chunk[a:b + 2]
            chunk = chunk[b + 2:]
            yield image


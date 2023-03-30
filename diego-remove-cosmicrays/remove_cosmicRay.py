from astropy.io import fits
import matplotlib.pyplot as plt
import astroscrappy
import numpy as np
import os

# Define la ruta de la carpeta que contiene las imágenes
ruta = '/Users/diegogonzalez/Dropbox/Mac/Documents/observaciones/ou5/spec/'

# Lee la lista de nombres de archivos en la carpeta
archivos = os.listdir(ruta)

# Recorre la lista de archivos y aplica la corrección de rayos cósmicos a cada imagen
for archivo in archivos:
    # Si el archivo no es un archivo fits, saltar al siguiente archivo
    if not archivo.endswith('.fits'):
        continue
    
    # Carga la imagen fits
    hdulist = fits.open(os.path.join(ruta, archivo))
    image = hdulist[0].data

    # Aplica la corrección de rayos cósmicos
    clean_image, mask = astroscrappy.detect_cosmics(image, sigclip=5.0, sigfrac=0.3, objlim=5.0, cleantype='median', verbose=False)

    # Calcula la imagen residual
    residual_image = image - clean_image

    # Convierte la imagen corregida y residual al tipo de datos original
    clean_image = clean_image.astype(np.float32)
    residual_image = residual_image.astype(np.float32)

    # Guarda la imagen corregida y la residual en nuevos archivos fits
    hdulist[0].data = clean_image
    hdulist.writeto(os.path.join(ruta, archivo.replace('.fits', '_c.fits')), overwrite=True)
    hdulist[0].data = residual_image
    hdulist.writeto(os.path.join(ruta, archivo.replace('.fits', '_res.fits')), overwrite=True)

    # Cierra el archivo fits
    hdulist.close()

    # Grafica la imagen original
    plt.figure(figsize=(10, 10))
    plt.imshow(image, vmin=np.percentile(image, 1), vmax=np.percentile(image, 99))
    plt.title('Imagen con rayos cósmicos')
    plt.colorbar()

    # Grafica la imagen corregida
    plt.figure(figsize=(10, 10))
    plt.imshow(clean_image, vmin=np.percentile(clean_image, 1), vmax=np.percentile(clean_image, 99))
    plt.title('Imagen sin rayos cósmicos')
    plt.colorbar()

    # Grafica la imagen residual
    plt.figure(figsize=(10, 10))
    plt.imshow(residual_image, vmin=np.percentile(residual_image, 1), vmax=np.percentile(residual_image, 99))
    plt.title('Imagen residual')
    plt.colorbar()
    
    plt.savefig('test.pdf')
    
    # Muestra las tres imágenes
    plt.show()
    




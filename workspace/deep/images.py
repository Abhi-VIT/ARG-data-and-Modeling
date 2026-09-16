import hashlib
import io
from pathlib import PurePosixPath
import stat
import warnings
import zipfile
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from ..ingestion import DataError

MAX_IMAGES=2000


def safe_image_name(name):
    path=PurePosixPath(name)
    if '\\' in name or ':' in name or path.is_absolute() or '..' in path.parts or not path.parts or len(name)>300:
        raise DataError('Image paths must be relative class-folder/file names without traversal.')
    return path


def read_images(path,progress=lambda percent,message:None):
    pixels=[];labels=[];names=[];seen={};duplicates=0
    try:
        with zipfile.ZipFile(path) as archive:
            infos=archive.infolist()
            if len(infos)>4000:raise DataError('Image archive contains too many entries (maximum 4,000).')
            total=0;files=[];paths=set()
            for info in infos:
                p=safe_image_name(info.filename)
                if stat.S_ISLNK(info.external_attr>>16) or info.flag_bits&1:raise DataError('Symlinks and encrypted ZIP entries are not supported.')
                if info.is_dir():continue
                if str(p) in paths:raise DataError('Image paths must be unique.')
                paths.add(str(p))
                total+=info.file_size
                if total>400*1024**2 or info.file_size>12*1024**2 or info.file_size/max(info.compress_size,1)>200:
                    raise DataError('Image archive exceeds expansion, per-file size, or compression-ratio limits.')
                if p.suffix.lower() not in {'.png','.jpg','.jpeg','.webp','.bmp'}:raise DataError('Image folders may contain only PNG, JPEG, WebP or BMP files.')
                files.append((info,p))
            if not 12<=len(files)<=MAX_IMAGES:raise DataError('Upload 12–2,000 images organized by class.')
            roots={p.parts[0] for _,p in files}
            strip_root=len(roots)==1 and all(len(p.parts)==3 for _,p in files)
            for i,(info,p) in enumerate(files):
                parts=p.parts[1:] if strip_root else p.parts
                if len(parts)!=2:raise DataError('Use class_name/image.jpg, optionally inside one common parent folder.')
                label=parts[0]
                content=archive.read(info)
                with warnings.catch_warnings():
                    warnings.simplefilter('error',Image.DecompressionBombWarning)
                    with Image.open(io.BytesIO(content)) as image:
                        expected={'.jpg':'JPEG','.jpeg':'JPEG','.png':'PNG','.webp':'WEBP','.bmp':'BMP'}[p.suffix.lower()]
                        if image.format!=expected or image.width*image.height>16_000_000 or max(image.size)>4096 or getattr(image,'n_frames',1)!=1:
                            raise DataError('Images must match their extension, be single-frame, and contain at most 16 megapixels / 4,096 pixels per side.')
                        image=ImageOps.exif_transpose(image).convert('RGB')
                        image=ImageOps.fit(image,(64,64),method=Image.Resampling.BILINEAR)
                        array=np.asarray(image,dtype=np.uint8).copy()
                digest=hashlib.sha256(array.tobytes()).hexdigest()
                if digest in seen:
                    if seen[digest]!=label:raise DataError('Identical images have conflicting class labels.')
                    duplicates+=1;continue
                seen[digest]=label;pixels.append(array.transpose(2,0,1));labels.append(label);names.append(str(p))
                if i%25==0:progress(15+int(75*i/len(files)),f'Validating image {i+1}/{len(files)}')
    except (zipfile.BadZipFile,UnidentifiedImageError,OSError,Image.DecompressionBombWarning,Image.DecompressionBombError) as exc:
        raise DataError('Invalid or unsafe image archive: '+str(exc)[:180]) from exc
    classes=sorted(set(labels))
    if not 2<=len(classes)<=20 or any(labels.count(c)<6 for c in classes):
        raise DataError('Use 2–20 classes with at least six distinct images per class after duplicate removal.')
    return np.stack(pixels),np.asarray([classes.index(c) for c in labels],dtype=np.int64),{'classes':classes,'names':names,'images':len(pixels),'duplicates_removed':duplicates,'image_size':64}

import os
import random
import json

PHOTO_VER = 'tx_000100000000'
SKETCH_VER = 'tx_000100000000'
RANDOM_SEED = 1234

if __name__ == '__main__':
    random.seed(RANDOM_SEED)
    photo_folder = os.path.join('./256x256/photo', PHOTO_VER)
    sketch_folder = os.path.join('./256x256/sketch', SKETCH_VER)
    # Obtenemos el listado de clases del dataset
    classes = os.listdir(photo_folder)
    # Creamos un diccionario para codificar las clases con números, class -> int
    class_dict = {}
    for i, name in enumerate(classes):
        class_dict[name] = i
    # Guardamos el diccionario en caso de necesitar decodificar las clases
    with open('./sketchy_classes.json', 'w') as f:
        json.dump(class_dict, f)
    # Seleccionamos las clases con las que vamos a entrenar
    random.shuffle(classes)
    known_classes = classes[:100]
    # Las que no se ocupan en entrenamiento se consideran desconocidas por el modelo
    unknown_classes = classes[100:]

    ## DATASET VALID_UNKNOWN
    valid_unknown_file = open('./valid_unknown.txt', 'w')
    # De las clases no conocidas seleccionamos 20 fotos y sus respectivos sketches (120 aprox)
    for current_class in unknown_classes:
        # Elegimos 20 fotos al azar
        photo_dir = os.path.join(photo_folder, current_class)
        selected_photos = random.sample(os.listdir(photo_dir), 20)
        # Buscamos en los sketches aquellos que corresponden a una foto elegida
        sketch_dir = os.path.join(sketch_folder, current_class)
        for sketch in os.listdir(sketch_dir):
            for photo in selected_photos:
                photo_no_ext, _ = os.path.splitext(os.path.basename(photo))
                # Si el sketch coincide a una foto entonces guardamos el par
                if photo_no_ext in sketch:
                    photo_abs = os.path.abspath(os.path.join(photo_dir, photo))
                    sketch_abs = os.path.abspath(os.path.join(sketch_dir, sketch))
                    valid_unknown_file.write(f'{sketch_abs}\t{photo_abs}\t{class_dict[current_class]}\n')
    valid_unknown_file.close()

    ## DATASET VALID_KNOWN y DATASET TRAIN
    train_file = open('./train.txt', 'w')
    valid_known_file = open('./valid_known.txt', 'w')
    for current_class in known_classes:
        # Separamos un 80% de las fotos para train y un 20% para validation
        photo_dir = os.path.join(photo_folder, current_class)
        photo_list = os.listdir(photo_dir)
        random.shuffle(photo_list)
        cutoff = int(len(photo_list)*0.8)
        train_photos = photo_list[:cutoff]
        valid_photos = photo_list[cutoff:]
        sketch_dir = os.path.join(sketch_folder, current_class)
        # Revisamos todos los sketches
        for sketch in os.listdir(sketch_dir):
            # Si coinciden con una foto de train, guardamos en train
            for photo in train_photos:
                photo_no_ext, _ = os.path.splitext(os.path.basename(photo))
                if photo_no_ext in sketch:
                    photo_abs = os.path.abspath(os.path.join(photo_dir, photo))
                    sketch_abs = os.path.abspath(os.path.join(sketch_dir, sketch))
                    train_file.write(f'{sketch_abs}\t{photo_abs}\t{class_dict[current_class]}\n')
            # Si coinciden con una foto de valid known, guardamos en valid known
            for photo in valid_photos:
                photo_no_ext, _ = os.path.splitext(os.path.basename(photo))
                if photo_no_ext in sketch:
                    photo_abs = os.path.abspath(os.path.join(photo_dir, photo))
                    sketch_abs = os.path.abspath(os.path.join(sketch_dir, sketch))
                    valid_known_file.write(f'{sketch_abs}\t{photo_abs}\t{class_dict[current_class]}\n')
    train_file.close()
    valid_known_file.close()

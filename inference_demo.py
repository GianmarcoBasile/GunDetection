import onnxruntime
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import argparse

class GunDetector:
    def __init__(self, model_path):
        """
        Inizializza il detector con il modello specificato.
        
        Args:
            model_path (str): Percorso al file del modello ONNX
        """
        self.model_path = model_path
        print(f"Caricamento del modello: {os.path.basename(model_path)}")
        self.session = onnxruntime.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.target_size = (150, 150)
        
    def preprocess_image(self, image):
        """
        Preprocess the input image for inference.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Preprocessed image as numpy array in NCHW format
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Resize diretto a target_size come nel training
        image = image.resize(self.target_size, Image.BILINEAR)
        
        # Converti in tensor e normalizza
        image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Espandi le dimensioni per il batch
        image = image.unsqueeze(0)
        
        # Converti in numpy per ONNX
        return image.numpy()
        
    def detect(self, image_path, threshold=0.3):
        """
        Esegue il rilevamento su un'immagine.
        
        Args:
            image_path (str): Percorso all'immagine da processare
            threshold (float): Soglia di confidenza per il rilevamento
            
        Returns:
            tuple: (immagine con risultati, boxes, scores, labels)
        """
        # Carica e preprocessa l'immagine
        original_image = Image.open(image_path).convert("RGB")
        original_size = original_image.size
        input_image = self.preprocess_image(original_image)
        
        try:
            # Esegui l'inferenza
            outputs = self.session.run(None, {self.input_name: input_image})
            
            boxes = outputs[0]
            scores = outputs[1]
            labels = outputs[2]
            
            if boxes is not None and isinstance(boxes, np.ndarray) and boxes.size > 0:
                # Riscala le box alle dimensioni originali
                ratio_x = original_size[0] / self.target_size[0]
                ratio_y = original_size[1] / self.target_size[1]
                
                scaled_boxes = boxes.copy()
                scaled_boxes[:, [0, 2]] *= ratio_x  # riscala x
                scaled_boxes[:, [1, 3]] *= ratio_y  # riscala y
                
                # Filtra per threshold
                mask = scores >= threshold
                scaled_boxes = scaled_boxes[mask]
                filtered_scores = scores[mask]
                filtered_labels = labels[mask]
                
                # Applica NMS
                if len(scaled_boxes) > 0:
                    # Calcola le aree e gli IoU
                    areas = (scaled_boxes[:, 2] - scaled_boxes[:, 0]) * (scaled_boxes[:, 3] - scaled_boxes[:, 1])
                    order = filtered_scores.argsort()[::-1]
                    keep = []
                    
                    while order.size > 0:
                        i = order[0]
                        keep.append(i)
                        
                        if order.size == 1:
                            break
                            
                        # Calcola IoU con il box corrente
                        xx1 = np.maximum(scaled_boxes[i, 0], scaled_boxes[order[1:], 0])
                        yy1 = np.maximum(scaled_boxes[i, 1], scaled_boxes[order[1:], 1])
                        xx2 = np.minimum(scaled_boxes[i, 2], scaled_boxes[order[1:], 2])
                        yy2 = np.minimum(scaled_boxes[i, 3], scaled_boxes[order[1:], 3])
                        
                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        
                        ovr = inter / (areas[i] + areas[order[1:]] - inter)
                        inds = np.where(ovr <= 0.1)[0]
                        order = order[inds + 1]
                    
                    # Mantieni solo i box selezionati
                    scaled_boxes = scaled_boxes[keep]
                    filtered_scores = filtered_scores[keep]
                    filtered_labels = filtered_labels[keep]
                
                print(f"\nModello: {self.model_path}")
                print(f"Forma dell'output: {[out.shape for out in outputs]}")
                for i, (box, score) in enumerate(zip(scaled_boxes, filtered_scores)):
                    print(f"Box {i+1}: Coordinate [x1={box[0]:.1f}, y1={box[1]:.1f}, x2={box[2]:.1f}, y2={box[3]:.1f}], Score: {score:.3f}")
                
                # Visualizza i risultati
                result_image = self.visualize_results(original_image, 
                                                   scaled_boxes,
                                                   filtered_scores,
                                                   filtered_labels,
                                                   threshold)
                
                return result_image, scaled_boxes, filtered_scores, filtered_labels
            
            return original_image, None, None, None
            
        except Exception as e:
            print(f"Errore durante l'inferenza: {str(e)}")
            return original_image, None, None, None
            
    def visualize_results(self, image, boxes, scores, labels, threshold=0.3):
        """
        Visualizza i risultati del rilevamento sull'immagine.
        
        Args:
            image: PIL Image
            boxes (np.ndarray): Box di rilevamento
            scores (np.ndarray): Score di confidenza
            labels (np.ndarray): Etichette delle classi
            threshold (float): Soglia di confidenza
            
        Returns:
            np.ndarray: Immagine con i box disegnati
        """
        # Converti l'immagine PIL in formato OpenCV
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Disegna i box con score superiore alla soglia
        if boxes is not None:
            if isinstance(boxes, np.ndarray):
                boxes = [boxes]
            if isinstance(scores, np.ndarray):
                scores = [scores]
                
            for box_list, score_list in zip(boxes, scores):
                if isinstance(box_list, np.ndarray) and box_list.size > 0:
                    for box, score in zip(box_list, score_list):
                        if score > threshold:
                            try:
                                x1, y1, x2, y2 = map(int, box)
                                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(image, f'Gun: {score:.2f}', (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Errore nel disegno del box: {str(e)}")
                                continue
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def nms(self, boxes, scores, iou_threshold=0.2):
        """
        Applica Non-Maximum Suppression ai box rilevati.
        
        Args:
            boxes (np.ndarray): Array di box in formato [x1, y1, x2, y2]
            scores (np.ndarray): Array di score di confidenza
            iou_threshold (float): Soglia IoU per considerare i box sovrapposti
            
        Returns:
            np.ndarray: Indici dei box da mantenere
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            # Calcola l'area di sovrapposizione relativa
            ovr = inter / np.minimum(areas[i], areas[order[1:]])
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]
            
        return np.array(keep)

def main():
    # Parser per gli argomenti della riga di comando
    parser = argparse.ArgumentParser(description='Esegui il rilevamento delle armi in un immagine.')
    parser.add_argument('image_path', type=str, help='Percorso all\'immagine da analizzare')
    args = parser.parse_args()

    # Percorsi dei modelli
    model_paths = [
        "Model/faster_rcnn.onnx",
        "Model/faster_rcnn_data_augmentation.onnx",
        "Model/faster_rcnn_higher_lr.onnx"
    ]
    
    if not os.path.exists(args.image_path):
        print(f"Immagine non trovata: {args.image_path}")
        return
    
    # Prepara il subplot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Confronto dei modelli")
    
    for idx, model_path in enumerate(model_paths):
        if not os.path.exists(model_path):
            print(f"Modello non trovato: {model_path}")
            continue
            
        try:
            # Inizializza il detector
            detector = GunDetector(model_path)
            
            # Esegui il rilevamento
            result_image, boxes, scores, labels = detector.detect(args.image_path)
            
            # Aggiungi al subplot
            axes[idx].imshow(result_image)
            axes[idx].set_title(os.path.basename(model_path).replace(".onnx", ""))
            axes[idx].axis('off')
                
        except Exception as e:
            print(f"Errore durante l'elaborazione del modello {model_path}: {str(e)}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 
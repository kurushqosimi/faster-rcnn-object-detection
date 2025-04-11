import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import VOCDetection
import torchvision.transforms as T
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Трансформации для изображений
transform = T.Compose([
    T.ToTensor(),
])

# Загрузка тренировочных данных Pascal VOC
dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)

# Функция для обработки данных в формат Faster R-CNN
def collate_fn(batch):
    return tuple(zip(*batch))

# DataLoader
data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Готовая модель Faster R-CNN с предварительными весами
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Количество классов (20 классов Pascal VOC + фон)
num_classes = 21 

# Меняем классификатор под наше количество классов
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Оптимизатор
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Цикл обучения
num_epochs = 2  # для демонстрации достаточно 2 эпох
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)
        
        # Подготовим целевые данные (bbox и labels)
        targets_formatted = []
        for target in targets:
            boxes = []
            labels = []
            obj = target['annotation']['object']
            if not isinstance(obj, list):
                obj = [obj]
            for o in obj:
                bbox = o['bndbox']
                box = [int(bbox['xmin']), int(bbox['ymin']), int(bbox['xmax']), int(bbox['ymax'])]
                boxes.append(box)
                labels.append(int(o['name'] == 'person'))  # пример: распознаем только людей (класс person = 1)

            boxes = torch.as_tensor(boxes, dtype=torch.float32).to(device)
            labels = torch.as_tensor(labels, dtype=torch.int64).to(device)
            
            targets_formatted.append({"boxes": boxes, "labels": labels})
        
        # Шаг обучения
        loss_dict = model(images, targets_formatted)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        epoch_loss += losses.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

print("🎉 Обучение завершено!")

model.eval()
img, _ = dataset[0]
with torch.no_grad():
    prediction = model([img.to(device)])

plt.imshow(img.permute(1,2,0))
ax = plt.gca()

for box, score in zip(prediction[0]['boxes'], prediction[0]['scores']):
    if score > 0.7:  
        xmin, ymin, xmax, ymax = box.cpu().numpy()
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

plt.title('Результаты Faster R-CNN')
plt.axis('off')
plt.show()


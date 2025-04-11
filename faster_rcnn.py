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

VOC_CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Оптимизатор
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Цикл обучения
num_epochs = 2  # можно увеличить потом, сейчас для теста 2 эпохи хватит
model.train()

for epoch in range(num_epochs):
    epoch_loss = 0
    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        targets_formatted = []
        for target in targets:
            boxes = []
            labels = []
            objs = target['annotation']['object']
            if not isinstance(objs, list):
                objs = [objs]
            for o in objs:
                bbox = o['bndbox']
                box = [
                    int(bbox['xmin']),
                    int(bbox['ymin']),
                    int(bbox['xmax']),
                    int(bbox['ymax'])
                ]
                boxes.append(box)

                # ВОТ ГЛАВНОЕ ИЗМЕНЕНИЕ!
                labels.append(VOC_CLASSES.index(o['name']))

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

print("🎉 Обучение завершено корректно на всех классах Pascal VOC!")

model.eval()

CLASSES = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

num_tests = 10
detected_objects = []

fig, axs = plt.subplots(2, 5, figsize=(20, 8))
axs = axs.flatten()

for img_idx in range(num_tests):
    img, _ = dataset[img_idx]
    with torch.no_grad():
        prediction = model([img.to(device)])

    axs[img_idx].imshow(img.permute(1, 2, 0))
    axs[img_idx].axis('off')
    axs[img_idx].set_title(f'Изображение {img_idx+1}')

    for obj_idx, (box, score, label) in enumerate(zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels'])):
        if score > 0.7:
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, 
                                 fill=False, color='red', linewidth=2)
            axs[img_idx].add_patch(rect)

            # Номер объекта в пределах одного изображения
            axs[img_idx].text(xmin, ymin - 10, f'{obj_idx+1}', color='yellow', fontsize=12, weight='bold')

            # Запоминаем детальнее объект
            class_name = CLASSES[label]
            detected_objects.append({
                'image_number': img_idx+1,
                'object_number': obj_idx+1,
                'class': class_name,
                'score': float(score.cpu().numpy())
            })

# Убираем пустые места, чтобы выглядело аккуратно
plt.tight_layout()
plt.show()

# Теперь выводим таблицу всех объектов
fig, ax = plt.subplots(figsize=(10, len(detected_objects)*0.5))
ax.axis('off')
plt.title('Список всех обнаруженных объектов', fontsize=14)

# Готовим красивый текст для вывода
text_info = '\n'.join([
    f"Изобр. {obj['image_number']}, Объект {obj['object_number']}: {obj['class']} ({obj['score']:.2f})"
    for obj in detected_objects
])

plt.text(0, 1, text_info, fontsize=12, verticalalignment='top')
plt.show()


# Сохраняем веса модели в файл fasterrcnn_model.pth
torch.save(model.state_dict(), 'fasterrcnn_model.pth')








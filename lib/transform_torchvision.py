def transform(self, image, label):
        # Resize
        resize = torchvision.transforms.Resize(size=(520, 520))
        image = resize(image)
        label = resize(label)

        # Random crop
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            image, output_size=(512, 512))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

        # Random horizontal flipping
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random vertical flipping
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # Transform to tensor
        image = TF.to_tensor(image).float()
        label = TF.to_tensor(label).long()
        return image, label
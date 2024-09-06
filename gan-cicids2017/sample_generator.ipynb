{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "import tensorflow as tf\n",
    "from keras import layers, models\n",
    "import matplotlib.pyplot as plt\n",
    "import os"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Load and preprocess the dataset\n",
    "def load_data(file_path):\n",
    "    # Load the dataset from a CSV file\n",
    "    data = pd.read_csv(file_path)\n",
    "    \n",
    "    # Drop non-numeric columns if any (e.g., labels, timestamps)\n",
    "    data = data.select_dtypes(include=[np.number])\n",
    "    \n",
    "    # Replace infinite values with NaN\n",
    "    data.replace([np.inf, -np.inf], np.nan, inplace=True)\n",
    "    \n",
    "    # Drop rows with NaN values\n",
    "    data.dropna(inplace=True)\n",
    "    \n",
    "    # Normalize the data to the range [-1, 1]\n",
    "    scaler = MinMaxScaler(feature_range=(-1, 1))\n",
    "    data = scaler.fit_transform(data)\n",
    "    \n",
    "    # Reshape the data if necessary (e.g., add channel dimension)\n",
    "    data = np.expand_dims(data, axis=-1)\n",
    "    \n",
    "    return data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# Define the generator model\n",
    "def build_generator(input_dim=100, output_shape=(78, 1)):\n",
    "    model = models.Sequential()\n",
    "    model.add(layers.Dense(256, input_dim=input_dim))\n",
    "    model.add(layers.LeakyReLU(alpha=0.2))\n",
    "    model.add(layers.BatchNormalization(momentum=0.8))\n",
    "    model.add(layers.Dense(512))\n",
    "    model.add(layers.LeakyReLU(alpha=0.2))\n",
    "    model.add(layers.BatchNormalization(momentum=0.8))\n",
    "    model.add(layers.Dense(1024))\n",
    "    model.add(layers.LeakyReLU(alpha=0.2))\n",
    "    model.add(layers.BatchNormalization(momentum=0.8))\n",
    "    model.add(layers.Dense(np.prod(output_shape), activation='tanh'))\n",
    "    model.add(layers.Reshape(output_shape))\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the discriminator model\n",
    "def build_discriminator(input_shape=(78, 1)):\n",
    "    model = models.Sequential()\n",
    "    model.add(layers.Flatten(input_shape=input_shape))\n",
    "    model.add(layers.Dense(512))\n",
    "    model.add(layers.LeakyReLU(alpha=0.2))\n",
    "    model.add(layers.Dense(256))\n",
    "    model.add(layers.LeakyReLU(alpha=0.2))\n",
    "    model.add(layers.Dense(1, activation='sigmoid'))\n",
    "    return model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Compile the models\n",
    "def compile_models(generator, discriminator):\n",
    "    discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])\n",
    "    discriminator.trainable = False\n",
    "    gan_input = layers.Input(shape=(100,))\n",
    "    generated_image = generator(gan_input)\n",
    "    gan_output = discriminator(generated_image)\n",
    "    gan = models.Model(gan_input, gan_output)\n",
    "    gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))\n",
    "    return gan"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save generated samples\n",
    "def save_samples(generator, epoch, output_dir='gan_samples', examples=5):\n",
    "    noise = np.random.normal(0, 1, (examples, 100))\n",
    "    generated_samples = generator.predict(noise)\n",
    "    generated_samples = 0.5 * generated_samples + 0.5 \n",
    "\n",
    "    if not os.path.exists(output_dir):\n",
    "        os.makedirs(output_dir)\n",
    "\n",
    "    for i in range(examples):\n",
    "        plt.figure(figsize=(10, 2))\n",
    "        plt.plot(generated_samples[i, :, 0])\n",
    "        plt.title(f'Sample {i + 1} at Epoch {epoch}')\n",
    "        plt.xlabel('Feature Index')\n",
    "        plt.ylabel('Value')\n",
    "        plt.savefig(f\"{output_dir}/gan_generated_sample_epoch_{epoch}_sample_{i + 1}.png\")\n",
    "        plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot training losses\n",
    "def plot_losses(d_losses, g_losses, output_dir='gan_samples'):\n",
    "    plt.figure(figsize=(10, 5))\n",
    "    plt.plot(d_losses, label='Discriminator Loss')\n",
    "    plt.plot(g_losses, label='Generator Loss')\n",
    "    plt.xlabel('Epoch')\n",
    "    plt.ylabel('Loss')\n",
    "    plt.legend()\n",
    "    plt.savefig(f\"{output_dir}/gan_losses.png\")\n",
    "    plt.close()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Train the GAN\n",
    "def train_gan(generator, discriminator, gan, data, epochs=10000, batch_size=64, save_interval=1000):\n",
    "    half_batch = batch_size // 2\n",
    "    d_losses = []\n",
    "    g_losses = []\n",
    "    \n",
    "    for epoch in range(epochs):\n",
    "        # Train discriminator\n",
    "        idx = np.random.randint(0, data.shape[0], half_batch)\n",
    "        real_samples = data[idx]\n",
    "        noise = np.random.normal(0, 1, (half_batch, 100))\n",
    "        fake_samples = generator.predict(noise)\n",
    "        d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))\n",
    "        d_loss_fake = discriminator.train_on_batch(fake_samples, np.zeros((half_batch, 1)))\n",
    "        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)\n",
    "        \n",
    "        # Train generator\n",
    "        noise = np.random.normal(0, 1, (batch_size, 100))\n",
    "        valid_y = np.array([1] * batch_size)\n",
    "        g_loss = gan.train_on_batch(noise, valid_y)\n",
    "        \n",
    "        # Save losses for plotting\n",
    "        d_losses.append(d_loss[0])\n",
    "        g_losses.append(g_loss)\n",
    "        \n",
    "        # Print progress\n",
    "        if epoch % 100 == 0:\n",
    "            print(f\"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]\")\n",
    "        \n",
    "        # Save generated samples at save_interval\n",
    "        if epoch % save_interval == 0:\n",
    "            save_samples(generator, epoch)\n",
    "    \n",
    "    # Plot the losses\n",
    "    plot_losses(d_losses, g_losses)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Main function to run the GAN\n",
    "def main():\n",
    "    file_path = '/home/cse/Documents/base-folder/DatasetToCheck/CICIDS2017/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'\n",
    "    data = load_data(file_path)\n",
    "    generator = build_generator()\n",
    "    discriminator = build_discriminator()\n",
    "    gan = compile_models(generator, discriminator)\n",
    "    train_gan(generator, discriminator, gan, data)\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    main()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

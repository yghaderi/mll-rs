use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
        Dropout, DropoutConfig, Linear,LinearConfig, Relu
    },
    prelude::*
};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem}
};

#[derive(Module, Debug)]
pub struct Model<B:Backend>{
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool: AdaptiveAvgPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu
}

#[derive(Config, Debug)]
pub struct ModelConfig{
    num_classes:usize,
    hidden_size: usize,
    #[config(default="0.5")]
    dropout: f64
}

impl ModelConfig{
    /// return the initialized model.
    pub fn init<B:Backend>(self, device: &B::Device)->Model<B>{
        Model{
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8,8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16*8*8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout:DropoutConfig::new(self.dropout).init()

        }
    }
}

impl <B:Backend> Model<B>{
    /// # Shapes
    ///   - Images [batch_size, height, width]
    ///   - Output [batch_size, num_classes]
    pub fn forward(&self, images: Tensor<B,3>)->Tensor<B,2>{
        let [batch_size, height, width] = images.dims();

        // Create a channel at the second dimension.
        let x = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(x); // [batch_size, 8, _, _]
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x= self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
        let x = x.reshape([batch_size, 16*8*8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x) // [batch_size, num_classes]
    }
}

#[derive(Clone)]
pub struct MnistBatcher<B: Backend>{
    device: B::Device,
}

impl <B:Backend> MnistBatcher<B>{
    pub fn new(device: B::Device)->Self{
        Self {device}
    }
}

pub struct MnistBatch<B: Backend>{
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>
}

pub

fn main() {
    println!("Hello, world!");
}

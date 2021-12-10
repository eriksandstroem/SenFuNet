import torch


class ConfidenceRouting(torch.nn.Module):
    """
    Confidence Routing Network
    """

    def __init__(self, Cin, F, batchnorms=True):

        super().__init__()
        self.F = F

        # for backwards compatibility to the old routing nets, set Cout = 2
        Cout = 1

        if batchnorms:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(Cout),
                torch.nn.ReLU(),
            )

            self.process = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(2 * F),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                torch.nn.BatchNorm2d(2 * F),
                torch.nn.ReLU(),
            )
        else:
            self.pre = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(Cin, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.post = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

            self.process = torch.nn.Sequential(
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(F, 2 * F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
                torch.nn.ReflectionPad2d(1),
                torch.nn.Conv2d(2 * F, 2 * F, kernel_size=3, stride=1, padding=0),
                torch.nn.ReLU(),
            )

        self.uncertainty = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3 * F, F, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(F, Cout, kernel_size=3, stride=1, padding=0),
            torch.nn.ReLU(),
        )

        self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, data):
        features = self.pre(data)
        lower_scale = self.maxpool(features)
        lower_features = self.process(lower_scale)
        upsampled = torch.nn.functional.interpolate(
            lower_features, scale_factor=2, mode="bilinear", align_corners=False
        )
        H = data.shape[2]
        W = data.shape[3]
        upsampled = upsampled[:, :, :H, :W]
        output = self.post(torch.cat((features, upsampled), dim=1))

        uncertainty = self.uncertainty(torch.cat((features, upsampled), dim=1))

        return torch.cat((output, uncertainty), dim=1)

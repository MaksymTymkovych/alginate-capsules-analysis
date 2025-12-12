
class Scale:
    def __init__(self, pixels_per_micron: float, unit: str = "µm"):
        self.pixels_per_micron = pixels_per_micron
        self.unit = unit
    def __repr__(self):
        return f"Scale(pixels_per_micron={self.pixels_per_micron}, unit='{self.unit}')"
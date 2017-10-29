from enum import Enum
from PyQt5.QtWidgets import QPushButton, QComboBox, QLabel, QProgressBar, QScrollArea


# kinda custom enum...
class Widgets:

    class Buttons(Enum):
        DatasetButton = "datasetButton"
        OutputDirectoryButton = "outputDirectoryButton"
        StartButton = "startButton"
        ImgDiffsButton = "imgDiffsButton"
        ClassifyButton = "classifyButton"
        ShowROCGraphs = "showROCGraphs"
        ShowPRGraphs = "showPRGraphs"

    class ComboBoxes(Enum):
        ResamplingAlgorithms = "resamplingAlgorithms"

    class Labels(Enum):
        DatasetPickedLabel = "datasetPickedLabel"
        DatasetStatisticsLabel = "datasetStatisticsLabel"
        ResamplingStatusLabel = "resamplingStatusLabel"
        ClassifyingStatusLabel = "classifyingStatusLabel"
        ResampledDatasetStatistics = "resampledDatasetStatistics"
        OutputDirectoryPickedLabel = "outputDirectoryPickedLabel"
        AfterClassificationStatistics = "afterClassificationStatistics"

    class ProgressBars(Enum):
        DatasetProgressBar = "datasetProgressBar"
        NormalClassifyProgressBar = "normalClassifyProgressBar"
        ResampleClassifyProgressBar = "resampleClassifyProgressBar"

    class ScrollAreas(Enum):
        AfterClassificationStatisticsArea = "afterClassificationStatisticsArea"

    def __init__(self, main_window):
        self.buttons = {button.value: main_window.findChild(QPushButton, button.value) for button in Widgets.Buttons}
        self.combo_boxes = {combo_box.value: main_window.findChild(QComboBox, combo_box.value)
                            for combo_box in Widgets.ComboBoxes}
        self.labels = {label.value: main_window.findChild(QLabel, label.value) for label in Widgets.Labels}
        self.progress_bars = {progress_bar.value: main_window.findChild(QProgressBar, progress_bar.value)
                             for progress_bar in Widgets.ProgressBars}
        self.scroll_areas = {scroll_area.value: main_window.findChild(QScrollArea, scroll_area.value)
                             for scroll_area in Widgets.ScrollAreas}

    def get_button(self, id):
        return self.buttons[id]

    def get_combo_box(self, id):
        return self.combo_boxes[id]

    def get_label(self, id):
        return self.labels[id]

    def get_progress_bar(self, id):
        return self.progress_bars[id]

    def get_scroll_area(self, id):
        return self.scroll_areas[id]
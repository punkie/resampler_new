from enum import Enum
from PyQt5.QtWidgets import QPushButton, QComboBox, QLabel, QProgressBar, QScrollArea, QTableWidget


# kinda custom enum...
class Widgets:
    def __init__(self, main_window):
        self.buttons = {button.value: main_window.findChild(QPushButton, button.value) for button in Widgets.Buttons}
        self.combo_boxes = {combo_box.value: main_window.findChild(QComboBox, combo_box.value)
                            for combo_box in Widgets.ComboBoxes}
        self.labels = {label.value: main_window.findChild(QLabel, label.value) for label in Widgets.Labels}
        self.progress_bars = {progress_bar.value: main_window.findChild(QProgressBar, progress_bar.value)
                              for progress_bar in Widgets.ProgressBars}
        self.scroll_areas = {scroll_area.value: main_window.findChild(QScrollArea, scroll_area.value)
                             for scroll_area in Widgets.ScrollAreas}
        self.tables = {table.value: main_window.findChild(QTableWidget, table.value) for table in Widgets.Tables}
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
        ClassificationAlgorithms = "classAlgorithms"

    class Labels(Enum):
        DatasetPickedLabel = "datasetPickedLabel"
        DatasetStatisticsLabel = "datasetStatisticsLabel"
        ResamplingStatusLabel = "resamplingStatusLabel"
        ClassifyingStatusLabel = "classifyingStatusLabel"
        ResampledDatasetStatistics = "resampledDatasetStatistics"
        OutputDirectoryPickedLabel = "outputDirectoryPickedLabel"
        AfterClassificationStatistics = "afterClassificationStatistics"
        ClassificationAlgorithmLabel = "classAlgorithmLabel"
        FilePathLabel = "filePathLabel"
        DatasetLoadingResultLabel = "datasetLoadingResultLabel"
        DatasetLoadingTextLabel = "datasetLoadingTextLabel"
        TotalNumberOfExamplesLabel = "totalNumberOfExamplesLabel"
        NumberOfNegativeExamplesLabel = "numberOfNegativeExamplesLabel"
        TargetClassPercentageLabel = "targetClassPercentageLabel"
        ImbalancedRatioLabel = "imbalancedRatioLabel"
        TotalNumberOfExamplesResampledLabel = "totalNumberOfExamplesResampledLabel"
        NumberOfNegativeExamplesResampledLabel = "numberOfNegativeExamplesResampledLabel"
        TargetClassPercentageResampledLabel = "targetClassPercentageResampledLabel"
        ImbalancedRatioResampledLabel = "imbalancedRatioResampledLabel"

    class ProgressBars(Enum):
        DatasetProgressBar = "datasetProgressBar"
        ResampleProgressBar = "resampleProgressBar"
        NormalClassifyProgressBar = "normalClassifyProgressBar"
        ResampleClassifyProgressBar = "resampleClassifyProgressBar"

    class ScrollAreas(Enum):
        AfterClassificationStatisticsArea = "afterClassificationStatisticsArea"

    class Tables(Enum):
        DataTable = "dataTableWidget"

    def get_button(self, widget_id):
        return self.buttons[widget_id]

    def get_combo_box(self, widget_id):
        return self.combo_boxes[widget_id]

    def get_label(self, widget_id):
        return self.labels[widget_id]

    def get_progress_bar(self, widget_id):
        return self.progress_bars[widget_id]

    def get_scroll_area(self, widget_id):
        return self.scroll_areas[widget_id]

    def get_table(self, widget_id):
        return self.tables[widget_id]
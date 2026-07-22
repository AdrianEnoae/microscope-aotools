import pathlib
import wx
from wx.lib.filebrowsebutton import DirBrowseButton

from microAO.aoRoutines import routines
from cockpit.util import logger

_REQUIRED_MLAO_ATTRS = ("TRIAL_MODES", "CORRECTION_MODES", "OFFSETS")

class fineTuningCollectionDialog(wx.Dialog):
    _REQUIRED_MLAO_ATTRS = ("TRIAL_MODES", "CORRECTION_MODES", "OFFSETS")

    def __init__(self, parent):
        super().__init__(
            parent,
            title="Fine Tuning Data Collection",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        )

        self._parent = parent
        self._device = parent._device

        panel = wx.Panel(self)

        self._routine_lookup = self._get_mlao_routine_lookup()
        routine_choices = list(self._routine_lookup.keys())

        if not routine_choices:
            raise RuntimeError("No MLAO-compatible routines were found.")

       

        self._combo_routine = wx.ComboBox(
            panel,
            value=routine_choices[0],
            choices=routine_choices,
            style=wx.CB_READONLY,
        )
        self._combo_routine.Bind(wx.EVT_COMBOBOX, self._on_routine_changed)
        self._text_target_modes = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self._text_bias_modes = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self._text_offsets = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self._text_num_bias_images = wx.TextCtrl(panel, style=wx.TE_READONLY)

        self._text_target_max_magnitude = wx.TextCtrl(panel, value="1.0")
        self._text_extra_modes = wx.TextCtrl(panel, value="")
        self._text_extra_max_magnitude = wx.TextCtrl(panel, value="0.0")
        self._text_batch_size = wx.TextCtrl(panel, value="32")

        self._dir_save_root = DirBrowseButton(
            panel,
            labelText="Save root:",
            startDirectory="",
        )

      
        widgets_data = (
            (wx.StaticText(panel, label="MLAO routine:"), wx.GBPosition(0, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._combo_routine, wx.GBPosition(0, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Target modes:"), wx.GBPosition(1, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_target_modes, wx.GBPosition(1, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Bias modes:"), wx.GBPosition(2, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_bias_modes, wx.GBPosition(2, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Bias offsets [rad]:"), wx.GBPosition(3, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_offsets, wx.GBPosition(3, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Images per datapoint:"), wx.GBPosition(4, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_num_bias_images, wx.GBPosition(4, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Target max magnitude [rad]:"), wx.GBPosition(5, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_target_max_magnitude, wx.GBPosition(5, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Extra random modes:"), wx.GBPosition(6, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_extra_modes, wx.GBPosition(6, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Extra max magnitude [rad]:"), wx.GBPosition(7, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_extra_max_magnitude, wx.GBPosition(7, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (wx.StaticText(panel, label="Batch size:"), wx.GBPosition(8, 0), wx.GBSpan(1, 1), wx.ALL, 5),
            (self._text_batch_size, wx.GBPosition(8, 1), wx.GBSpan(1, 1), wx.ALL | wx.EXPAND, 5),

            (self._dir_save_root, wx.GBPosition(9, 0), wx.GBSpan(1, 2), wx.ALL | wx.EXPAND, 5),
        )

        panel_sizer = wx.GridBagSizer(vgap=0, hgap=0)
        panel_sizer.SetCols(2)
        panel_sizer.AddGrowableCol(1)
        for widget_data in widgets_data:
            panel_sizer.Add(*widget_data)
        panel.SetSizer(panel_sizer)

        self._button_run = wx.Button(self, label="Run Collection")
        self._button_run.Bind(wx.EVT_BUTTON, self._on_run)

        button_cancel = wx.Button(self, wx.ID_CANCEL, label="Cancel")

        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        button_sizer.Add(self._button_run, 0, wx.ALL, 5)
        button_sizer.Add(button_cancel, 0, wx.ALL, 5)

        root_sizer = wx.BoxSizer(wx.VERTICAL)
        root_sizer.Add(panel, 1, wx.EXPAND)
        root_sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 5)

        self.SetSizerAndFit(root_sizer)

        self._update_routine_fields()


    def _get_mlao_routine_lookup(self):
        routine_lookup = {}

        for routine_key, routine_cls in routines.items():
            try:
                defaults = routine_cls.defaults()
            except Exception:
                continue

            if defaults.get("type") != "MLAO":
                continue

            if not all(hasattr(routine_cls, attr) for attr in _REQUIRED_MLAO_ATTRS):
                continue

            routine_lookup[routine_cls.name()] = routine_key

        return routine_lookup

    def _routine_label_from_key(self, routine_key):
        if routine_key is None:
            return None

        for label, key in self._routine_lookup.items():
            if key == routine_key:
                return label

        return None

    def _get_selected_routine(self):
        routine_label = self._combo_routine.GetValue()
        routine_key = self._routine_lookup[routine_label]
        routine_cls = routines[routine_key]
        return routine_key, routine_cls

    def _on_routine_changed(self, event):
        del event
        self._update_routine_fields()

    def _update_routine_fields(self):
        _, routine_cls = self._get_selected_routine()

        target_modes = list(routine_cls.CORRECTION_MODES)
        bias_modes = list(routine_cls.TRIAL_MODES)
        offsets = list(routine_cls.OFFSETS)
        num_bias_images = len(bias_modes) * len(offsets)

        self._text_target_modes.SetValue(self._modes_to_text(target_modes))
        self._text_bias_modes.SetValue(self._modes_to_text(bias_modes))
        self._text_offsets.SetValue(", ".join(str(x) for x in offsets))
        self._text_num_bias_images.SetValue(str(num_bias_images))

    def _on_run(self, event):
        del event

        params = self._parse_params()
        if params is None:
            return

        camera = self._parent.getCamera()
        if camera is None:
            return

        imager = self._parent.getImager()
        if imager is None:
            return

        self._button_run.Disable()

        try:
            self._device.collectFineTuningData(camera, imager, params)
            self.EndModal(wx.ID_OK)
        except Exception as exc:
            self._button_run.Enable()
            logger.log.error(
                "Fine-tuning data collection failed: {}".format(exc)
            )
            wx.MessageBox(
                "Fine-tuning data collection failed:\n{}".format(exc),
                caption="Error",
            )

    def _parse_params(self):
        try:
            routine_key, routine_cls = self._get_selected_routine()

            target_max_magnitude = float(
                self._text_target_max_magnitude.GetValue()
            )

            extra_modes = self._parse_modes(
                self._text_extra_modes.GetValue(),
                allow_empty=True,
                label="extra modes",
            )

            extra_max_magnitude = float(
                self._text_extra_max_magnitude.GetValue()
            )

            batch_size = int(self._text_batch_size.GetValue())

        except ValueError as exc:
            self._show_parse_error(str(exc))
            return None

        if target_max_magnitude <= 0:
            self._show_parse_error(
                "Target max magnitude must be greater than 0 radians."
            )
            return None

        if extra_max_magnitude < 0:
            self._show_parse_error(
                "Extra max magnitude must be 0 or greater."
            )
            return None

        if extra_max_magnitude > 0 and len(extra_modes) == 0:
            self._show_parse_error(
                "Extra max magnitude is greater than 0, but no extra modes are selected."
            )
            return None

        if batch_size < 1:
            self._show_parse_error("Batch size must be at least 1.")
            return None

        save_root_value = self._dir_save_root.GetValue().strip()
        if not save_root_value:
            self._show_parse_error("A save root directory must be selected.")
            return None

        return {
            "routine": routine_key,
            "routine_name": routine_cls.name(),
            "target_modes": list(routine_cls.CORRECTION_MODES),
            "trial_modes": list(routine_cls.TRIAL_MODES),
            "offsets": list(routine_cls.OFFSETS),
            "target_max_magnitude_rad": target_max_magnitude,
            "extra_modes": extra_modes,
            "extra_max_magnitude_rad": extra_max_magnitude,
            "batch_size": batch_size,
            "save_root": pathlib.Path(save_root_value),
            }

    def _show_parse_error(self, message):
        with wx.MessageDialog(
            self,
            message,
            "Parsing error",
            wx.OK | wx.ICON_ERROR,
        ) as dlg:
            dlg.ShowModal()

    @staticmethod
    def _parse_modes(text, allow_empty, label):
        text = text.strip()
        if not text:
            if allow_empty:
                return []
            raise ValueError("{} cannot be empty.".format(label.capitalize()))

        modes = []
        parts = [part.strip() for part in text.split(",") if part.strip()]

        for part in parts:
            if "-" in part:
                bounds = [bound.strip() for bound in part.split("-")]
                if len(bounds) != 2:
                    raise ValueError("Could not parse {}: {}".format(label, part))

                start = int(bounds[0])
                stop = int(bounds[1])

                if stop < start:
                    raise ValueError("Invalid mode range in {}: {}".format(label, part))

                modes.extend(range(start, stop + 1))
            else:
                modes.append(int(part))

        if any(mode < 0 for mode in modes):
            raise ValueError(
                "{} must be zero-based correction-vector indices.".format(
                    label.capitalize()
                )
            )

        return list(dict.fromkeys(modes))

    @staticmethod
    def _modes_to_text(modes):
        return ", ".join(str(mode) for mode in modes)
import wx
from OpenGL.GL import *
from cockpit.gui.camera.viewPanel import ViewPanel, VIEW_WIDTH, VIEW_HEIGHT
from cockpit import depot
from cockpit.gui.imageViewer.viewCanvas import ViewCanvas, HISTOGRAM_HEIGHT, DRAG_ROI, DRAG_CANVAS, DRAG_BLACKPOINT, DRAG_WHITEPOINT
from cockpit import events
from cockpit.gui.guiUtils import placeMenuAtMouse

class ROIViewCanvas(ViewCanvas):
    """Subclass of the original ViewCanvas that redirects the ROI selection from the camera to the sensorlessAO routine"""
    def onMouse(self, event):
        if self.imageShape is None:
            return
        self.curMouseX, self.curMouseY = (event.GetPosition() *
                                          self.GetContentScaleFactor())
        self.updateMouseInfo(self.curMouseX, self.curMouseY)
        if event.LeftDClick():
            # Explicitly skip EVT_LEFT_DCLICK for parent to handle.
            event.ResumePropagation(2)
            event.Skip()
        elif event.LeftDown():
            # Started dragging
            self.mouseDragX, self.mouseDragY = self.curMouseX, self.curMouseY
            self.mouseLdownX, self.mouseLdownY = self.curMouseX, self.curMouseY
            blackPointX = 0.5 * (1+self.histogram.data2gl(self.histogram.lthresh)) * self.w
            whitePointX = 0.5 * (1+self.histogram.data2gl(self.histogram.uthresh)) * self.w
            # Set drag mode based on current window position
            if self.h - self.curMouseY >= (HISTOGRAM_HEIGHT *
                                           self.GetContentScaleFactor()* 2):
                if self.definingROI:
                    self.dragMode = DRAG_ROI
                else:
                    self.dragMode = DRAG_CANVAS
            elif abs(self.curMouseX - blackPointX) < abs(self.curMouseX - whitePointX):
                self.dragMode = DRAG_BLACKPOINT
            else:
                self.dragMode = DRAG_WHITEPOINT
        elif event.LeftIsDown():
            # Drag mouse. Different behaviors depending on drag mode.
            if self.dragMode == DRAG_CANVAS:
                # Pan view about.
                # Window coordinates are upside-down compared to what the
                # user expects.
                self.modPan(self.curMouseX - self.mouseDragX,
                            self.mouseDragY - self.curMouseY)
            elif self.dragMode in [DRAG_BLACKPOINT, DRAG_WHITEPOINT]:
                glx = -1 + 2 * self.curMouseX / self.w
                threshold = self.histogram.gl2data(glx)
                if self.dragMode == DRAG_BLACKPOINT:
                    self.histogram.lthresh = threshold
                    self.image.vmin = threshold
                else:
                    self.histogram.uthresh = threshold
                    self.image.vmax = threshold
            elif self.dragMode == DRAG_ROI:
                # Get co-ordinates in canvas units
                coords_x = [self.mouseDragX, self.mouseLdownX]
                coords_y = [self.mouseDragY, self.mouseLdownY]
                roi_xmin, roi_ymin = min(coords_x), min(coords_y)
                roi_xmax, roi_ymax = max(coords_x), max(coords_y)

                # Convert to data indices           
                roi_min_ind = self.canvasToIndices(roi_xmin, roi_ymin)
                roi_max_ind = self.canvasToIndices(roi_xmax, roi_ymax)

                # Get size of roi
                roi_maxsize = max((roi_max_ind[0] - roi_min_ind[0], roi_max_ind[1] - roi_min_ind[1]))
                roi_size = (roi_maxsize, roi_maxsize)

                # Set roi (left, top, width, height)
                self.roi_drag = (roi_min_ind[1], roi_min_ind[0], roi_size[1], roi_size[0]) 

            self.mouseDragX = self.curMouseX
            self.mouseDragY = self.curMouseY
        elif event.RightDown():
            placeMenuAtMouse(self, self._menu)
        elif event.LeftUp():
            #Changed here from original
            if self.definingROI:
                self.roi_drag=list(self.roi_drag)

                if self.roi_drag[2]<256:
                    print('ROI must be at least 256 pixels wide')
                    self.roi=None
                else:
                    if self.roi_drag[2]%2==1:
                        self.roi_drag[2]==self.roi_drag[2]-1
                        self.roi_drag[3]==self.roi_drag[3]-1

                camera = self.Parent.Parent.curCamera
                camera_roi=camera.getROI()
                self.roi_drag[0]=self.roi_drag[0]+camera_roi[0]
                self.roi_drag[1]=self.roi_drag[1]+camera_roi[1]

                self.roi = tuple(self.roi_drag)
                self.definingROI = False

        elif event.Entering() and self.TopLevelParent.IsActive():
            self.SetFocus()
        else:
            event.Skip()

        # In case current mouse position has changed enough to require
        # redrawing the histogram. A bit wasteful of resources, this.
        wx.CallAfter(self.Refresh)

    #Ovewritting the paint function to make the ROI always display
    def onPaint(self, event):
        if not self.shouldDraw:
            return

        try:
            # Unused, but wx requires we create an instance of PaintDC.
            dc = wx.PaintDC(self)
        except:
            return

        if not self.haveInitedGL:
            self.InitGL()

        if self.painting:
            print("Concurrent - returning")
            return

        try:
            Hist_Height=int(HISTOGRAM_HEIGHT*self.GetContentScaleFactor())
            self.painting = True
            self.SetCurrent(self.context)
            glClear(GL_COLOR_BUFFER_BIT)
            glViewport(0, Hist_Height,
                       self.w, self.h - Hist_Height)
            self.image.draw(pan=(self.panX, self.panY), zoom=self.zoom)
            if self.showCrosshair:
                self.drawCrosshair()

            self.drawROI()

            glViewport(0, 0, self.w, Hist_Height//2)
            self.histogram.draw()
            glColor(0, 1, 0, 1)

            glViewport(0, 0, self.w, self.h)
            glMatrixMode (GL_PROJECTION)
            glPushMatrix()
            glLoadIdentity ()
            glOrtho (0, self.w, 0, self.h, 1., -1.)
            glTranslatef(0, (Hist_Height)/2+2, 0)
            try:
                self.face.render('%d [%-10d %10d] %d' %
                                 (self.image.dmin, self.histogram.lthresh,
                                  self.histogram.uthresh,
                                  self.image.dmin+self.image.dptp))
            except:
                pass
            glPopMatrix()

            #self.drawHistogram()

            #glFlush()
            self.SwapBuffers()
            self.drawEvent.set()
        except Exception as e:
            print ("Error drawing view canvas:",e)
            traceback.print_exc()
            #self.shouldDraw = False
        finally:
            self.painting = False

    def getMenuActions(self):
        return [
                ('Set ROI', self.onDefineROI),
                ('Clear ROI', self.onClearROI)
                ]

    def onClearROI(self, event = None):
        self.roi = None

    def getROI(self):
        if self.roi:
            return self.roi
        else:
            return None
  

    


class ROIViewPanel(ViewPanel):
    """A ViewPanel that uses ROIViewCanvas instead of the default."""
    def enable(self, camera):
        self.selector.SetLabel(camera.descriptiveName)
        self.selector.SetBackgroundColour(camera.color)
        self.selector.Refresh()
        self.curCamera = camera

        # NB the 512 here is the largest texture size our graphics card can
        # gracefully handle.
        self.canvas = ROIViewCanvas(self.canvasPanel,
                                    size = (VIEW_WIDTH, VIEW_HEIGHT))
        self.canvas.SetSize((VIEW_WIDTH, VIEW_HEIGHT))
        self.canvas.resetView()

        # Subscribe to new image events only after canvas is prepared.
        events.subscribe(events.NEW_IMAGE % self.curCamera.name, self.onImage)




class ROISelectionDialog(wx.Dialog):
    def __init__(self, parent):
        super().__init__(
            parent,
            title="ROI Setup",
            style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        )
        self._camera=self.getCamera()
        self._ROI=None
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self._view_panel = ROIViewPanel(self) 
        main_sizer.Add(self._view_panel, 1, wx.EXPAND | wx.ALL, 5)

        if self._camera:
            self._view_panel.enable(self._camera)
        else:
            # If no camera is passed, the panel will just show "No camera"
            print("Warning: No camera provided to ROISelectionDialog")

        self.Bind(wx.EVT_CLOSE, self._on_close)
        snap_button = wx.Button(self, label="Snap Image")
        snap_button.Bind(wx.EVT_BUTTON, self._on_snap_button_click)
        main_sizer.Add(snap_button, 0, wx.ALL | wx.CENTER, 5)

        

        sizer_stdbuttons = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        main_sizer.Add(sizer_stdbuttons, 0, wx.ALIGN_RIGHT | wx.ALL, 5)
        self.SetSizerAndFit(main_sizer)

        self.Bind(wx.EVT_CLOSE, self._on_close)
        self.Bind(wx.EVT_BUTTON, self._on_ok, id=wx.ID_OK)
        self.Bind(wx.EVT_BUTTON, self._on_cancel, id=wx.ID_CANCEL)


    def _on_snap_button_click(self, event):
        wx.CallAfter(wx.GetApp().Imager.takeImage)  

    def getSelectedROI(self):
        if self._ROI:
            return self._ROI
        else:
            return None
    
    def _on_ok(self, event):
        if self._view_panel:
            self._ROI=self._view_panel.canvas.getROI()
            self._view_panel.disable()
        self.EndModal(wx.ID_OK)

    def _on_cancel(self, event):
        if self._view_panel:
            self._view_panel.disable()
        self.EndModal(wx.ID_CANCEL)

    def _on_close(self,event):
        if self._view_panel:
            self._view_panel.disable()
        self.EndModal(wx.ID_CANCEL)



    def getCamera(self):
        cameras = depot.getActiveCameras()

        camera = None
        if not cameras:
            wx.MessageBox(
                "There are no cameras enabled.", caption="No cameras active"
            )
        elif len(cameras) == 1:
            camera = cameras[0]
        else:
            cameras_dict = dict([(camera.descriptiveName, camera) for camera in cameras])

            with wx.SingleChoiceDialog(
                None,
                "Select camera",
                "Camera",
                list(cameras_dict.keys()),
                wx.CHOICEDLG_STYLE
            ) as dlg:
                if dlg.ShowModal() == wx.ID_OK:
                    camera = cameras_dict[dlg.GetStringSelection()]

        return camera
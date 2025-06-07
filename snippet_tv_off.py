# Snippet from: tv_off.py
# Generated for client preview - Full code available after payment
# Contact for complete implementation

    def __init__(self):
        super().__init__()
        self.start_time = time.time()
        
        # Enhanced effect parameters
        self.speed_factor = 1.0
        self.horizontal_chromatic = 1.0
        self.vertical_chromatic = 1.0
        self.dramatic_mode = True  # New parameter for enhanced effects
        
        # Enhanced timing variables for more dramatic effect
        self.fps = 60.0
        self.frame_time = 1000.0 / self.fps
        self.period = int(1500.0 / self.frame_time)  # Longer cycle: 45 frames (1.5s)
        self.burst_duration = int(300.0 / self.frame_time)  # Longer burst: 9 frames (0.3s)
        self.static_duration = int(200.0 / self.frame_time)  # Static phase: 6 frames (0.2s)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def initializeGL(self):
        print("Initializing TV Off Effect...")
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        
        try:
            # Compile shaders
            self.prog0 = compileProgram(
                compileShader(PASS0_VERT, GL_VERTEX_SHADER),
                compileShader(PASS0_FRAG, GL_FRAGMENT_SHADER),
            )
            self.prog1 = compileProgram(
                compileShader(PASS1_VERT, GL_VERTEX_SHADER),
                compileShader(PASS1_FRAG, GL_FRAGMENT_SHADER),
            )
            self.prog2 = compileProgram(
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def loadInitialTexture(self):
        """Load initial texture - try input.png first, then create test pattern"""
        try:
            if os.path.exists('input.png'):
                self.tex_input = self.loadTexture('input.png')
                self.current_image_path = 'input.png'
                print("input.png loaded")
            else:
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def setupQuad(self):
        positions = [
            -1.0, -1.0, 0.0,  0.0, 0.0,  # Bottom-left
             1.0, -1.0, 0.0,  1.0, 0.0,  # Bottom-right
             1.0,  1.0, 0.0,  1.0, 1.0,  # Top-right
            -1.0,  1.0, 0.0,  0.0, 1.0,  # Top-left
        ]
        
        self.vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def setupFramebuffers(self):
        self.fbo = glGenFramebuffers(1)
        self.pass0_tex = self.createEmptyTexture()
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def loadTexture(self, path):
        """Load texture from file path with error handling"""
        try:
            # Open and convert image
            img = Image.open(path)
            
            # Convert to RGBA if not already
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
            
            # Get image data
            data = img.tobytes("raw", "RGBA", 0, -1)
            
            # Create OpenGL texture
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def loadNewImage(self, file_path):
        """Load a new image and replace the current texture"""
        try:
            # Delete old texture if it exists and isn't the test pattern
            if hasattr(self, 'tex_input') and self.current_image_path is not None:
                glDeleteTextures(1, [self.tex_input])
            
            # Load new texture
            self.tex_input = self.loadTexture(file_path)
            self.current_image_path = file_path
            
            print(f"Successfully loaded new image: {file_path}")
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def createTestPattern(self):
        """Create a colorful test pattern if no input image"""
        size = 512
        data = []
        for y in range(size):
            for x in range(size):
                r = int(255 * (x / size))
                g = int(255 * (y / size))
                b = int(255 * ((x + y) / (2 * size)))
                data.extend([r, g, b, 255])
        
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def createEmptyTexture(self):
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        for tex in [self.pass0_tex, self.pass1_tex]:
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def drawQuad(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.index_buffer)
        
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 5*4, ctypes.c_void_p(0))
        
        glEnableVertexAttribArray(1)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def updateTiming(self):
        """Enhanced timing with more dramatic phases"""
        elapsed = time.time() - self.start_time
        total_frames = int(elapsed * self.fps * self.speed_factor)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def paintGL(self):
        self.updateTiming()
        w, h = self.width(), self.height()
        
        # Enhanced phase detection
        is_burst_phase = self.current_index < self.burst_duration
        is_static_phase = (self.current_index >= self.burst_duration and 
                          self.current_index < self.burst_duration + self.static_duration)
        is_shutoff_phase = self.current_index >= self.burst_duration + self.static_duration
        
        # PASS 0 - Enhanced Chromatic Aberration
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.pass0_tex, 0)
        glViewport(0, 0, w, h)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.prog0)
        
        # Enhanced chromatic aberration with more dramatic timing
        if is_burst_phase:
            # More dramatic chromatic aberration during burst
            burst_progress = self.current_index / self.burst_duration
            intensity_curve = math.sin(burst_progress * math.pi * 4) * (1 - burst_progress * 0.5)
            texel_offset = intensity_curve * 300.0  # Much stronger effect
        elif is_static_phase:
            # Flickering chromatic during static phase
            texel_offset = (random.random() - 0.5) * 150.0
        else:
            texel_offset = 0.0
            
        glUniform1f(glGetUniformLocation(self.prog0, 'u_iwidthoffset'), 
                   self.horizontal_chromatic / w * 2.0)  # Increased scale
        glUniform1f(glGetUniformLocation(self.prog0, 'u_iheightoffset'), 
                   self.vertical_chromatic / h * 2.0)  # Increased scale
        glUniform1f(glGetUniformLocation(self.prog0, 'u_texeloffset'), texel_offset)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.tex_input)
        glUniform1i(glGetUniformLocation(self.prog0, 'inputImageTexture'), 0)
        
        self.drawQuad()
        
        # PASS 1 - Enhanced Wave Distortion + TV Effects
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.pass1_tex, 0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glUseProgram(self.prog1)
        
        if is_burst_phase:
            # Enhanced burst phase with more dramatic effects
            burst_progress = self.current_index / self.burst_duration
            intensity = math.sin(burst_progress * math.pi * 2) * (1 - burst_progress * 0.3)
            
            u_xscale = random.uniform(0.05, 0.25) * intensity  # Much stronger waves
            u_yscale = random.uniform(0.03, 0.15) * intensity
            u_black = 0
            u_tv_shutoff = 0.0
            u_scanline = 1.0
            u_noise = intensity * 0.4  # Add noise during burst
            u_flicker = intensity * 0.8  # Strong flicker
            
        elif is_static_phase:
            # Static/noise phase
            u_xscale = random.uniform(-0.02, 0.02)  # Subtle random movement
            u_yscale = random.uniform(-0.02, 0.02)
            u_black = 0
            u_tv_shutoff = 0.0
            u_scanline = 1.0
            u_noise = 0.8  # Heavy static
            u_flicker = 0.6  # Moderate flicker
            
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def __init__(self):
        super().__init__()
        self.setWindowTitle("TV Off Effect - CapCut Recreation")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget with effect
        self.effect_widget = TVOffEffect()
        self.setCentralWidget(self.effect_widget)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def create_menu(self):
        """Create menu bar with file operations"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Load image action
        load_action = QtWidgets.QAction('Load Image...', self)
        load_action.setShortcut('Ctrl+O')
        load_action.setStatusTip('Load an image file')
        load_action.triggered.connect(self.load_image_dialog)
        file_menu.addAction(load_action)
        
        # Reset to test pattern action
        reset_action = QtWidgets.QAction('Reset to Test Pattern', self)
        reset_action.setStatusTip('Reset to default test pattern')
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def load_image_dialog(self):
        """Open file dialog to load an image"""
        file_dialog = QtWidgets.QFileDialog()
        file_dialog.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        file_dialog.setNameFilter("Image files (*.png *.jpg *.jpeg *.bmp *.tiff *.gif);;All files (*.*)")
        file_dialog.setWindowTitle("Load Image for TV Off Effect")
        
    # Full code available after Monday payment confirmation
    # Contact for complete implementation


    def load_image(self, file_path):
        """Load an image file"""
        try:
            self.effect_widget.loadNewImage(file_path)
            filename = os.path.basename(file_path)
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def reset_to_test_pattern(self):
        """Reset to the default test pattern"""
        try:
            # Delete current texture if it exists
            if hasattr(self.effect_widget, 'tex_input') and self.effect_widget.current_image_path is not None:
                glDeleteTextures(1, [self.effect_widget.tex_input])
            
            # Create new test pattern
            self.effect_widget.tex_input = self.effect_widget.createTestPattern()
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def create_controls(self):
        # Create dock widget for controls
        dock = QtWidgets.QDockWidget("Controls", self)
        dock.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        
        controls = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(controls)
        
        # Image loading section
        image_group = QtWidgets.QGroupBox("Image")
        image_layout = QtWidgets.QVBoxLayout(image_group)
        
        load_btn = QtWidgets.QPushButton("Load Image...")
        load_btn.clicked.connect(self.load_image_dialog)
        image_layout.addWidget(load_btn)
        
        reset_btn = QtWidgets.QPushButton("Reset to Test Pattern")
        reset_btn.clicked.connect(self.reset_to_test_pattern)
        image_layout.addWidget(reset_btn)
        
        layout.addWidget(image_group)
        
        # Effect controls section
        effect_group = QtWidgets.QGroupBox("Effect Parameters")
        effect_layout = QtWidgets.QVBoxLayout(effect_group)
        
        # Speed control
        effect_layout.addWidget(QtWidgets.QLabel("Speed:"))
        speed_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        speed_slider.setRange(10, 200)
        speed_slider.setValue(100)  # Default 1.0
        speed_slider.valueChanged.connect(self.on_speed_changed)
        effect_layout.addWidget(speed_slider)
        
        # Horizontal chromatic
        effect_layout.addWidget(QtWidgets.QLabel("Horizontal Chromatic:"))
        h_chrom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        h_chrom_slider.setRange(0, 200)
        # Full code available after Monday payment confirmation
        # Contact for complete implementation


    def on_speed_changed(self, value):
        self.effect_widget.speed_factor = value / 100.0
        
    # Full code available after Monday payment confirmation


    def on_h_chromatic_changed(self, value):
        self.effect_widget.horizontal_chromatic = value / 100.0 * 5.0  # Scale to reasonable range
        
    # Full code available after Monday payment confirmation


    def on_v_chromatic_changed(self, value):
        self.effect_widget.vertical_chromatic = value / 100.0 * 5.0  # Scale to reasonable range

    # Full code available after Monday payment confirmation


    def dragEnterEvent(self, event):
        """Handle drag enter events for file dropping"""
        if event.mimeData().hasUrls():
            event.accept()
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


    def dropEvent(self, event):
        """Handle file drop events"""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            # Take the first file
            file_path = files[0]
            # Check if it's an image file
            # Full code available after Monday payment confirmation
            # Contact for complete implementation


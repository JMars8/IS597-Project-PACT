import tkinter as tk
from tkinter import ttk


class ChatGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Simple Chat Interface")
        self.geometry("700x500")

        # Settings state
        self.setting_a_enabled = tk.BooleanVar(value=False)
        self.setting_b_choice = tk.StringVar(value="1")
        self.setting_c_choice = tk.StringVar(value="1")

        # Configure overall grid: output on top (expand), input at bottom (fixed)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.columnconfigure(0, weight=1)

        self._build_output_area()
        self._build_input_area()
        self._build_menu_bar()

    def _build_output_area(self) -> None:
        """Scrollable text area for conversation history."""
        frame = ttk.Frame(self)
        frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=(8, 4))

        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.output_text = tk.Text(
            frame,
            wrap="word",
            state="disabled",
            bg="white",
            relief="solid",
            borderwidth=1,
        )

        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)

        self.output_text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

    def _build_menu_bar(self) -> None:
        """Create a simple menu bar with a Settings item."""
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        settings_menu = tk.Menu(menubar, tearoff=False)
        settings_menu.add_command(label="Settings…", command=self.open_settings_window)
        menubar.add_cascade(label="Options", menu=settings_menu)

    def _build_input_area(self) -> None:
        """Multi-line input box and send button at the bottom."""
        frame = ttk.Frame(self)
        frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 8))

        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0)

        # Settings gear button on the left
        settings_button = ttk.Button(frame, text="⚙", width=3, command=self.open_settings_window)
        settings_button.grid(row=0, column=0, sticky="w", padx=(0, 4))

        self.input_text = tk.Text(
            frame,
            height=3,
            wrap="word",
            relief="solid",
            borderwidth=1,
        )
        self.input_text.grid(row=0, column=1, sticky="ew", padx=(0, 4))

        send_button = ttk.Button(frame, text="Send", command=self.on_send_clicked)
        send_button.grid(row=0, column=2, sticky="e")

        # Bind Enter to send, Shift+Enter for newline
        self.input_text.bind("<Return>", self._on_enter_pressed)
        self.input_text.bind("<Shift-Return>", self._on_shift_enter_pressed)

    def _on_enter_pressed(self, event: tk.Event) -> str:
        self.on_send_clicked()
        return "break"  # Prevent newline on plain Enter

    def _on_shift_enter_pressed(self, event: tk.Event) -> None:
        # Allow a normal newline when Shift+Enter is pressed
        self.input_text.insert("insert", "\n")

    def on_send_clicked(self) -> None:
        user_message = self.input_text.get("1.0", "end-1c").strip()
        if not user_message:
            return

        # Clear input box
        self.input_text.delete("1.0", "end")

        # For now, just respond with "thank you"
        response = "thank you"

        self._append_message("You", user_message)
        self._append_message("Bot", response)

    def _append_message(self, sender: str, message: str) -> None:
        self.output_text.configure(state="normal")
        self.output_text.insert("end", f"{sender}: {message}\n\n")
        self.output_text.see("end")
        self.output_text.configure(state="disabled")

    def open_settings_window(self) -> None:
        """Open a small settings window with Set A, B, C."""
        win = tk.Toplevel(self)
        win.title("Settings")
        win.transient(self)
        win.grab_set()

        # Layout
        padding = {"padx": 10, "pady": 6}

        # Set A: On/Off
        ttk.Label(win, text="Set A:").grid(row=0, column=0, sticky="w", **padding)
        ttk.Checkbutton(
            win,
            text="Enabled",
            variable=self.setting_a_enabled,
        ).grid(row=0, column=1, sticky="w", **padding)

        # Set B: radio 1 / 2
        ttk.Label(win, text="Set B:").grid(row=1, column=0, sticky="w", **padding)
        b_frame = ttk.Frame(win)
        b_frame.grid(row=1, column=1, sticky="w", **padding)
        ttk.Radiobutton(
            b_frame,
            text="1",
            value="1",
            variable=self.setting_b_choice,
        ).pack(side="left")
        ttk.Radiobutton(
            b_frame,
            text="2",
            value="2",
            variable=self.setting_b_choice,
        ).pack(side="left")

        # Set C: radio 1 / 2
        ttk.Label(win, text="Set C:").grid(row=2, column=0, sticky="w", **padding)
        c_frame = ttk.Frame(win)
        c_frame.grid(row=2, column=1, sticky="w", **padding)
        ttk.Radiobutton(
            c_frame,
            text="1",
            value="1",
            variable=self.setting_c_choice,
        ).pack(side="left")
        ttk.Radiobutton(
            c_frame,
            text="2",
            value="2",
            variable=self.setting_c_choice,
        ).pack(side="left")

        # Close button
        btn_frame = ttk.Frame(win)
        btn_frame.grid(row=3, column=0, columnspan=2, sticky="e", padx=10, pady=(0, 10))
        ttk.Button(btn_frame, text="Close", command=win.destroy).pack(side="right")


if __name__ == "__main__":
    app = ChatGUI()
    app.mainloop()

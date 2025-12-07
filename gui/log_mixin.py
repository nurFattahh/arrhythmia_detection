class LogMixin:
    def log(self, text):
        try:
            self.log_box.append(str(text))
            cursor = self.log_box.textCursor()
            self.log_box.setTextCursor(cursor)
            self.log_box.ensureCursorVisible()

            max_lines = 100
            lines = self.log_box.toPlainText().splitlines()
            if len(lines) > max_lines:
                self.log_box.setPlainText("\n".join(lines[-max_lines:]))
        except Exception:
            # silent fallback
            print(text)
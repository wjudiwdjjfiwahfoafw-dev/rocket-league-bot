#include "KeyPressDetector.h"

#ifdef _MSC_VER
#include <conio.h>
#include <io.h>

char GGL::KeyPressDetector::GetPressedChar() {
	return _getch();
}

bool GGL::KeyPressDetector::IsInteractive() {
	return _isatty(_fileno(stdin)) != 0;
}

#else
#include <unistd.h>
#include <termios.h>
#include <cstdlib>

// The process usually exits while the key thread is blocked in read() with echo/canonical mode turned off,
//	which leaves the user's shell without echo. So we remember the original settings and restore them at exit.
static struct termios g_OrigTermios = {};
static bool g_OrigTermiosSaved = false;

static void RestoreTerminal() {
	if (g_OrigTermiosSaved)
		tcsetattr(STDIN_FILENO, TCSANOW, &g_OrigTermios);
}

char GGL::KeyPressDetector::GetPressedChar() {
	// https://stackoverflow.com/questions/421860/capture-characters-from-standard-input-without-waiting-for-enter-to-be-pressed
	char buf = 0;
	struct termios old = { 0 };
	if (tcgetattr(0, &old) < 0)
		perror("tcsetattr()");
	else if (!g_OrigTermiosSaved) {
		g_OrigTermios = old;
		g_OrigTermiosSaved = true;
		atexit(RestoreTerminal);
	}
	old.c_lflag &= ~ICANON;
	old.c_lflag &= ~ECHO;
	old.c_cc[VMIN] = 1;
	old.c_cc[VTIME] = 0;
	if (tcsetattr(0, TCSANOW, &old) < 0)
		perror("tcsetattr ICANON");
	if (read(0, &buf, 1) < 0)
		perror("read()");
	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;
	if (tcsetattr(0, TCSADRAIN, &old) < 0)
		perror("tcsetattr ~ICANON");
	return (buf);
}

bool GGL::KeyPressDetector::IsInteractive() {
	return isatty(STDIN_FILENO) != 0;
}

#endif
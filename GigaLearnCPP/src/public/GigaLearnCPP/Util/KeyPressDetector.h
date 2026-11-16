#pragma once

#include "../Framework.h"

namespace GGL {
	namespace KeyPressDetector {
		char GetPressedChar();

		// True if stdin is an interactive terminal that key presses can be read from
		// (false when running headless, e.g. under nohup/systemd/docker or with stdin redirected)
		bool IsInteractive();
	}
}
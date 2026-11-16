#pragma once
#include "Report.h"
#include <pybind11/pybind11.h>

namespace GGL {
	struct RG_IMEXPORT MetricSender {
		std::string curRunID;
		std::string projectName, groupName, runName;
		pybind11::module pyMod;

		MetricSender(std::string projectName = {}, std::string groupName = {}, std::string runName = {}, std::string runID = {});
		
		RG_NO_COPY(MetricSender);

		void Send(const Report& report);

		// Call when the run is over, so the receiver can upload its remaining metrics and close the run
		void Finish();

		~MetricSender();
	};
}
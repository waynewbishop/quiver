# watchOS Guide

Analyzing sessions, measuring them geometrically, and fitting models from Apple Watch telemetry.

## Overview

Building a personal baseline is the primary goal for statistics and machine learning on the wrist. A baseline represents everything the watch knows about its wearer. It includes typical pace and resting heart rate and movement signatures. We use statistics to describe this baseline while linear algebra measures how a new session compares to it. Machine learning then fits a model that reflects the history of the wearer.

This same pattern works across every athletic discipline from running and cycling to swimming and strength training. We turn data from the wrist into a model of the wearer and compute it directly on the watch. This ensures that the data stays on the device and remains owned by the wearer at all times.

> Tip: To see these pieces assembled into one working model (a personal baseline, a residual read against it, and an effort classifier, all built from the primitives this guide surveys), see <doc:Building-An-Effort-Model>, the worked example this guide accompanies.

### Setup and lifecycle

A watchOS app computes directly on the wrist and keeps the data of a wearer on the device. Sensor readings arrive as a stream during a workout while the app turns each window of that stream into the plain numbers Quiver works with. This conversion happens at the edge so that every value is already a simple array by the time it reaches a statistic or a model.

What the watch learns should also outlive the workout that taught it. A baseline or a fitted classifier or a stored set of feature vectors is worth saving when a session ends. We read this data back when the next session begins so that the model of the wearer deepens over time rather than starting fresh each run. Every Quiver value is built to be saved and restored without ceremony and to move safely between the concurrent tasks of a live session. The sections below explore the values that flow through this lifecycle while the guide for model persistence covers the save and restore shape they all share.


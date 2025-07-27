#
"""
Interactive Matplotlib plot with a snapping cursor.

This script displays a plot of a sine curve. As you move your mouse over the plot,
a crosshair cursor snaps to the nearest data point on the curve, and the coordinates
of that point are displayed in the upper right corner. The crosshair and text disappear
when the mouse leaves the plot area.

Usage:
    Run the script. Move your mouse over the plot to see the snapping cursor in action.

Reference: https://matplotlib.org/3.5.0/gallery/misc/cursor_demo.html
"""
import numpy as np
import matplotlib.pyplot as plt

class SnappingCursor:    
    def __init__(self, ax, line):
        self.ax = ax
        self.horizontal_line = ax.axhline(color='k', lw=0.8, ls='--')
        self.vertical_line = ax.axvline(color='k', lw=0.8, ls='--')
        self.data_x, self.data_y = line.get_data()
        self._last_index = None
        # text location in axes coords
        self.text = ax.text(0.72, 0.9, '', transform=ax.transAxes)

    def set_cross_hair_visible(self, visible):
        need_redraw = self.horizontal_line.get_visible() != visible
        self.horizontal_line.set_visible(visible)
        self.vertical_line.set_visible(visible)
        self.text.set_visible(visible)
        return need_redraw

    def on_mouse_move(self, event):
        fig = self.ax.figure
        # Hide crosshair if mouse is outside axes or data is invalid
        if not event.inaxes:
            self._last_index = None
            if self.set_cross_hair_visible(False):
                fig.canvas.draw()
            return
        # Snap to nearest data point
        self.set_cross_hair_visible(True)
        cursor_x, cursor_y = event.xdata, event.ydata
        index = min(np.searchsorted(self.data_x, cursor_x), len(self.data_x) - 1)
        if index == self._last_index:
            return  # still on the same data point
        self._last_index = index
        cursor_x = self.data_x[index]
        cursor_y = self.data_y[index]
        # Update crosshair and text
        self.horizontal_line.set_ydata([cursor_y, cursor_y])
        self.vertical_line.set_xdata([cursor_x, cursor_x])
        self.text.set_text(f'x={cursor_x:.2f}, y={cursor_y:.2f}')
        fig.canvas.draw()

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6, 4))
    # draw plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    line, = ax.plot(x, y)
    # setup cursor
    snap_cursor = SnappingCursor(ax, line)
    fig.canvas.mpl_connect('motion_notify_event', snap_cursor.on_mouse_move)
    # start
    plt.show()
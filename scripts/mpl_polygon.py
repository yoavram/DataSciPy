"""
This script allows interactive drawing of polygons on a matplotlib plot.
Left-click to start drawing, and release to close the polygon.
If the polygon contains any points from the plotted line, it is colored red.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class PolygonDrawer:
    def __init__(self, ax, line):
        self.ax = ax
        self.line_data = np.array(line.get_data())

    
    def mouse_handler(self, event):
        if not event.dblclick and event.button == 1 and event.inaxes:
            ax = event.inaxes
            if event.name == 'button_press_event':
                self.path = ax.plot([], [], 'k-')[0]
            xdata, ydata = self.path.get_data()
            x, y = event.xdata, event.ydata            
            xdata = np.append(xdata, x)
            ydata = np.append(ydata, y)
            if event.name == 'button_release_event':
                xdata = np.append(xdata, xdata[0])
                ydata = np.append(ydata, ydata[0])
            self.path.set_data(xdata, ydata)
            if event.name == 'button_release_event':                
                path_data = np.array(self.path.get_data())
                self.poly = plt.Polygon(path_data.T, alpha=0.3)   
                if self.poly.contains_points(self.line_data.T).any():
                    self.poly.set_color('r')
                ax.add_artist(self.poly)
            self.ax.figure.canvas.draw()

if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(6, 4))
    # draw plot
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    line, = ax.plot(x, y)
    # setup polygon drawer
    polygon_drawer = PolygonDrawer(ax, line)
    for event in ('button_press_event', 'button_release_event', 'motion_notify_event'):
        fig.canvas.mpl_connect(event, polygon_drawer.mouse_handler)
    # start
    plt.show()    
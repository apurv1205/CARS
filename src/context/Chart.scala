package context

import java.awt.Color

import org.jfree.chart.axis.{AxisLocation, NumberAxis}
import org.jfree.chart.labels.StandardCategoryToolTipGenerator
import org.jfree.chart.plot.{DatasetRenderingOrder, PlotOrientation}
import org.jfree.chart.renderer.category.LineAndShapeRenderer
import org.jfree.chart.{ChartFactory, ChartFrame}
import org.jfree.data.category.DefaultCategoryDataset

/**
  * Created by roger19890107 on 4/5/16.
  */
object Chart {
  def plotBarLineChart(title: String, xLabel: String, yBarLabel: String,
                       yBarMin: Double, yBarMax: Double, yLineLabel: String,
                       dataBarChart: DefaultCategoryDataset,
                       dataLineChart: DefaultCategoryDataset) = {

    // build bar chart
    val chart = ChartFactory.createBarChart(
      "", xLabel, yBarLabel, dataBarChart,
      PlotOrientation.VERTICAL, true, true, false)

    // get plot
    val plot = chart.getCategoryPlot
    plot.setBackgroundPaint(new Color(0xEE, 0xEE, 0xFF))
    plot.setDomainAxisLocation(AxisLocation.BOTTOM_OR_RIGHT)
    plot.setDataset(1, dataLineChart)

    // bar chart y axis
    val vn = plot.getRangeAxis()
    vn.setRange(yBarMin, yBarMax)

    // line chart y axis
    val axis2 = new NumberAxis(yLineLabel)
    plot.setRangeAxis(1, axis2)

    // set line chart in front of bar chart
    val renderer2 = new LineAndShapeRenderer()
    renderer2.setToolTipGenerator(new StandardCategoryToolTipGenerator())
    plot.setRenderer(1, renderer2)
    plot.setDatasetRenderingOrder(DatasetRenderingOrder.FORWARD)

    // build frame
    val frame = new ChartFrame(title, chart)
    frame.setSize(500, 500)
    frame.pack()
    frame.setVisible(true)
  }
}

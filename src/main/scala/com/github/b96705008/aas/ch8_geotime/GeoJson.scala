package com.github.b96705008.aas.ch8_geotime

import com.esri.core.geometry.{Geometry, GeometryEngine}
import spray.json._

import scala.collection.mutable.ArrayBuffer

/**
  * Created by roger19890107 on 7/23/16.
  */

case class Feature(id: Option[JsValue],
                   properties: Map[String, JsValue],
                   geometry: RichGeometry) {
  def apply(property: String) = properties(property)
  def get(property: String) = properties.get(property)
}

case class FeatureCollection(features: Array[Feature])
  extends IndexedSeq[Feature] {
  def apply(index: Int) = features(index)
  def length = features.length
}

case class GeometryCollection(geometries: Array[RichGeometry])
  extends IndexedSeq[RichGeometry] {
  def apply(index: Int) = geometries(index)
  def length = geometries.length
}

/**
  * GeoJsonProtocol for transform between obj and JSON
  */
object GeoJsonProtocol extends DefaultJsonProtocol {
  // RichGeometry
  implicit object RichGeometryJsonFormat extends RootJsonFormat[RichGeometry] {
    def write(g: RichGeometry) = {
      GeometryEngine.geometryToJson(g.spatialReference, g.geometry).parseJson
    }

    def read(value: JsValue) = {
      val mg = GeometryEngine.geometryFromGeoJson(value.compactPrint, 0, Geometry.Type.Unknown)
      new RichGeometry(mg.getGeometry, mg.getSpatialReference)
    }
  }

  // Feature
  implicit object FeatureJsonFormat extends RootJsonFormat[Feature] {
    def write(f: Feature) = {
      val buf = ArrayBuffer(
        "type" -> JsString("Feature"),
        "properties" -> JsObject(f.properties),
        "geometry" -> f.geometry.toJson
      )
      f.id.foreach(v => buf += "id" -> v)
      JsObject(buf.toMap)
    }

    def read(value: JsValue) = {
      val jso = value.asJsObject
      val id = jso.fields.get("id")
      val properties = jso.fields("properties").asJsObject.fields
      val geometry = jso.fields("geometry").convertTo[RichGeometry]
      Feature(id, properties, geometry)
    }
  }

  // Feature Collection
  implicit object FeatureCollectionJsonFormat extends RootJsonFormat[FeatureCollection] {
    def write(fc: FeatureCollection) = {
      JsObject(
        "type" -> JsString("FeatureCollection"),
        "features" -> JsArray(fc.features.map(_.toJson): _*) // to multiple params
      )
    }

    def read(value: JsValue) = {
      FeatureCollection(value.asJsObject.fields("features").convertTo[Array[Feature]])
    }
  }

  // Geometry Collection
  implicit object GeometryCollectionJsonFormat extends RootJsonFormat[GeometryCollection] {
    def write(gc: GeometryCollection) = {
      JsObject(
        "type" -> JsString("GeometryCollection"),
        "geometries" -> JsArray(gc.geometries.map(_.toJson): _*)
      )
    }

    def read(value: JsValue) = {
      GeometryCollection(value.asJsObject.fields("geometries").convertTo[Array[RichGeometry]])
    }
  }
}



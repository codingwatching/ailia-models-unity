using ailiaSDK;
using System.Collections.Generic;
using System;
using UnityEngine;

using ailia;

namespace ailiaSDK
{
	public class AiliaRenderer3D : MonoBehaviour
	{
		public GameObject spheres; //Spheres
		public GameObject sphere; //Sphere to instiate
		public List<GameObject> sphereObjectBuffer = new List<GameObject>();
		public int sphereObjectBufferIndex = 0;

		public GameObject lines3D;
		public GameObject line3D;
		List<GameObject> lineObjectBuffer3D = new List<GameObject>();
		int lineObjectBufferIndex3D = 0;

		// 座標系の描画用
		public List<GameObject> axisSphereObjectBuffer = new List<GameObject>();
		public int axisSphereObjectBufferIndex = 0;
		List<GameObject> axisLineObjectBuffer3D = new List<GameObject>();
		int axisLineObjectBufferIndex3D = 0;

		public Vector3 drawPos = new Vector3(0.0f, 0.0f, 120.0f); //3次元landmarkの原点
		public float scale = 20; //3次元landmarkの原点の倍率
		public float time = 0; //3次元landmarkを回転させるために実行してからの経過時間を保持する

		private List<uint> LANDMARK_LEFT = new List<uint>() { 1, 3, 5, 7, 9, 11, 13, 15 };
		private List<uint> LANDMARK_RIGHT = new List<uint>() { 2, 4, 6, 8, 10, 12, 14, 16 };

		public void Clear()
		{
			for (int i = sphereObjectBufferIndex; i < sphereObjectBuffer.Count; i++)
			{
				sphereObjectBuffer[i].SetActive(false);
			}
			sphereObjectBufferIndex = 0;

			for (int i = lineObjectBufferIndex3D; i < lineObjectBuffer3D.Count; i++)
			{
				lineObjectBuffer3D[i].SetActive(false);
			}
			lineObjectBufferIndex3D = 0;

			for (int i = axisSphereObjectBufferIndex; i < axisSphereObjectBuffer.Count; i++)
			{
				axisSphereObjectBuffer[i].SetActive(false);
			}
			axisSphereObjectBufferIndex = 0;

			for (int i = axisLineObjectBufferIndex3D; i < axisLineObjectBuffer3D.Count; i++)
			{
				axisLineObjectBuffer3D[i].SetActive(false);
			}
			axisLineObjectBufferIndex3D = 0;
		}

		public void DrawBone3D(Color32 color, AiliaPoseEstimator.AILIAPoseEstimatorObjectPose obj, uint from, uint to)
		{
			float th = 0.1f;
			if (obj.points[from].score <= th || obj.points[to].score <= th)
			{
				return;
			}

			float from_x = obj.points[from].x;
			float from_y = obj.points[from].y;
			float from_z = obj.points[from].z_local;
			float to_x = obj.points[to].x;
			float to_y = obj.points[to].y;
			float to_z = obj.points[to].z_local;
			float origin_x = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].x + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].x) / 2.0f;
			float origin_y = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].y + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].y) / 2.0f;
			float origin_z = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].z_local + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].z_local) / 2.0f;

			//腰の中点を原点に変える
			from_x -= origin_x;
			from_y -= origin_y;
			from_z -= origin_z;
			to_x -= origin_x;
			to_y -= origin_y;
			to_z -= origin_z;

			//3次元landmarkを回転させる
			double speed = Math.PI / 100.0f; //1フレームあたりの回転角
			time += Time.deltaTime;
			float tmp_from_x = from_x; //更新前の値
			float tmp_from_y = from_y;
			float tmp_from_z = from_z;
			float tmp_to_x = to_x;
			float tmp_to_y = to_y;
			float tmp_to_z = to_z;
			from_x = (float)(tmp_from_x * Math.Cos(speed / Math.PI * time) + tmp_from_z * Math.Sin(speed / Math.PI * time));
			from_y = tmp_from_y;
			from_z = (float)(-tmp_from_x * Math.Sin(speed / Math.PI * time) + tmp_from_z * Math.Cos(speed / Math.PI * time));
			to_x = (float)(tmp_to_x * Math.Cos(speed / Math.PI * time) + tmp_to_z * Math.Sin(speed / Math.PI * time));
			to_y = tmp_to_y;
			to_z = (float)(-tmp_to_x * Math.Sin(speed / Math.PI * time) + tmp_to_z * Math.Cos(speed / Math.PI * time));

			Color sphereColor = Color.white; //球の色だけここで指定する 右(231, 217, 0)
            if (LANDMARK_LEFT.Contains(from))
            {
				sphereColor = new Color(0.0f, 179.0f / 255, 255.0f / 255, 1.0f); //左
			}
			else if (LANDMARK_RIGHT.Contains(from))
            {
				sphereColor = new Color(248.0f / 255, 123.0f / 255, 0.0f, 1.0f); //右
			}

			DrawSphere3D(sphereColor, from_x, from_y, from_z);
			DrawLine3D(color, from_x, from_y, from_z, to_x, to_y, to_z);
		}

		public void DrawSphere3D(Color color, float pos_x, float pos_y, float pos_z)
		{
			GameObject newSphere;
			if (sphereObjectBufferIndex < sphereObjectBuffer.Count)
			{
				newSphere = sphereObjectBuffer[sphereObjectBufferIndex];
			}
			else
			{
				newSphere = Instantiate(sphere, spheres.gameObject.transform);
				sphereObjectBuffer.Add(newSphere);
			}
			sphereObjectBufferIndex++;
			newSphere.SetActive(true);

			MeshRenderer mesh = newSphere.GetComponent<MeshRenderer>(); //球の色を変更
			mesh.material.color = color;

			Vector3 pointPos = drawPos; //原点の座標 見えやすいところにずらす
			pointPos.x += pos_x * scale;
			pointPos.y += pos_y * scale;
			pointPos.z += pos_z * scale;

			newSphere.transform.position = pointPos; //大きさと位置を調整
		}

		public void DrawLine3D(Color32 color, float from_x, float from_y, float from_z, float to_x, float to_y, float to_z)
		{
			Vector3 pointPos1 = drawPos; //原点の座標
			pointPos1.x += from_x * scale;
			pointPos1.y += from_y * scale;
			pointPos1.z += from_z * scale;

			Vector3 pointPos2 = drawPos; //原点の座標
			pointPos2.x += to_x * scale;
			pointPos2.y += to_y * scale;
			pointPos2.z += to_z * scale;

			GameObject newLine;
			LineRenderer lRend;
			if (lineObjectBufferIndex3D < lineObjectBuffer3D.Count)
			{
				newLine = lineObjectBuffer3D[lineObjectBufferIndex3D];
				lRend = newLine.GetComponent<LineRenderer>();
			}
			else
			{
				newLine = Instantiate(line3D, lines3D.gameObject.transform);
				newLine.layer = line3D.gameObject.layer;
				lRend = newLine.GetComponent<LineRenderer>();
				lineObjectBuffer3D.Add(newLine);
			}
			lineObjectBufferIndex3D++;
			newLine.SetActive(true);

			Color32 c1 = color;
			c1.a = 128 + 32;

			lRend.startColor = c1;
			lRend.endColor = c1;

			lRend.positionCount = 2;
			lRend.startWidth = 1; //仮
			lRend.endWidth = 1; //仮

			Vector3 startVec = pointPos1;
			Vector3 endVec = pointPos2;
			lRend.SetPosition(0, startVec);
			lRend.SetPosition(1, endVec);
		}

		public void DrawAxis3D(AiliaPoseEstimator.AILIAPoseEstimatorObjectPose obj)
		{
			float scale = 1.0f;

			//腰の中点を原点にする（DrawBone3Dと同じ座標系にする）
			float origin_x = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].x + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].x) / 2.0f;
			float origin_y = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].y + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].y) / 2.0f;
			float origin_z = (obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_LEFT].z_local + obj.points[AiliaPoseEstimator.AILIA_POSE_ESTIMATOR_POSE_KEYPOINT_HIP_RIGHT].z_local) / 2.0f;

			//全てのlandmarkのx,y,z座標のうち，絶対値が最大のものを取得する
			//また，y座標が最大のものを取得する（画面上でY正が下方向のため、y_maxが足元=地面）
			float abs_max = 0.0f;
			float y_max = -1.0f;
			for (int i = 0; i < obj.points.Length; i++)
			{
				float cx = obj.points[i].x - origin_x;
				float cy = obj.points[i].y - origin_y;
				float cz = obj.points[i].z_local - origin_z;
				abs_max = Mathf.Max(abs_max, Math.Abs(cx), Math.Abs(cy), Math.Abs(cz));
				y_max = Mathf.Max(y_max, cy);
			}
			scale = abs_max + 0.1f;

			//外側の軸（グリッドをy_maxに配置、ボックスをy_max - scale*2まで上方向に伸ばす＝人がボックスの中に入る）
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), -scale, y_max, -scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), -scale, y_max, scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), -scale, y_max - scale * 2, -scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), -scale, y_max - scale * 2, scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), scale, y_max, -scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), scale, y_max, scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), scale, y_max - scale * 2, -scale);
			DrawAxisSphere3D(new Color(0.0f, 92.0f / 255, 0.0f, 1.0f), scale, y_max - scale * 2, scale);
			DrawAxisLine3D(Color.white, -scale, y_max, -scale, scale, y_max, -scale);
			DrawAxisLine3D(Color.white, -scale, y_max - scale * 2, -scale, scale, y_max - scale * 2, -scale);
			DrawAxisLine3D(Color.white, -scale, y_max - scale * 2, scale, scale, y_max - scale * 2, scale);
			DrawAxisLine3D(Color.white, -scale, y_max, scale, scale, y_max, scale);
			DrawAxisLine3D(Color.white, -scale, y_max, -scale, -scale, y_max - scale * 2, -scale);
			DrawAxisLine3D(Color.white, scale, y_max, -scale, scale, y_max - scale * 2, -scale);
			DrawAxisLine3D(Color.white, scale, y_max, scale, scale, y_max - scale * 2, scale);
			DrawAxisLine3D(Color.white, -scale, y_max, scale, -scale, y_max - scale * 2, scale);
			DrawAxisLine3D(Color.white, -scale, y_max, -scale, -scale, y_max, scale);
			DrawAxisLine3D(Color.white, scale, y_max, -scale, scale, y_max, scale);
			DrawAxisLine3D(Color.white, scale, y_max - scale * 2, -scale, scale, y_max - scale * 2, scale);
			DrawAxisLine3D(Color.white, -scale, y_max - scale * 2, -scale, -scale, y_max - scale * 2, scale);

			//内側のグリッド線
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, -scale * 0.75f, scale, y_max, -scale * 0.75f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, -scale * 0.5f, scale, y_max, -scale * 0.5f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, -scale * 0.25f, scale, y_max, -scale * 0.25f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, 0.0f, scale, y_max, 0.0f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, scale * 0.25f, scale, y_max, scale * 0.25f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, scale * 0.5f, scale, y_max, scale * 0.5f, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale, y_max, scale * 0.75f, scale, y_max, scale * 0.75f, 0.5f);

			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale * 0.75f, y_max, -scale, -scale * 0.75f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale * 0.5f, y_max, -scale, -scale * 0.5f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), -scale * 0.25f, y_max, -scale, -scale * 0.25f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), 0.0f, y_max, -scale, 0.0f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), scale * 0.25f, y_max, -scale, scale * 0.25f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), scale * 0.5f, y_max, -scale, scale * 0.5f, y_max, scale, 0.5f);
			DrawAxisLine3D(new Color(1.0f, 1.0f, 1.0f, 0.1f), scale * 0.75f, y_max, -scale, scale * 0.75f, y_max, scale, 0.5f);

		}

		public void DrawAxisSphere3D(Color color, float pos_x, float pos_y, float pos_z)
		{
			GameObject newSphere;
			if (axisSphereObjectBufferIndex < axisSphereObjectBuffer.Count)
			{
				newSphere = axisSphereObjectBuffer[axisSphereObjectBufferIndex];
			}
			else
			{
				newSphere = Instantiate(sphere, spheres.gameObject.transform);
				axisSphereObjectBuffer.Add(newSphere);
			}
			axisSphereObjectBufferIndex++;
			newSphere.SetActive(true);

			MeshRenderer mesh = newSphere.GetComponent<MeshRenderer>();
			mesh.material.color = color;

			//回転させる
			double speed = Math.PI / 100.0f; //1フレームあたりの回転角
			time += Time.deltaTime;
			float tmp_x = pos_x;
			float tmp_y = pos_y;
			float tmp_z = pos_z;
			pos_x = (float)(tmp_x * Math.Cos(speed / Math.PI * time) + tmp_z * Math.Sin(speed / Math.PI * time));
			pos_y = tmp_y;
			pos_z = (float)(-tmp_x * Math.Sin(speed / Math.PI * time) + tmp_z * Math.Cos(speed / Math.PI * time));

			Vector3 pointPos = drawPos; //大きさと位置を調整
			pointPos.x += pos_x * this.scale;
			pointPos.y += pos_y * this.scale;
			pointPos.z += pos_z * this.scale;

			newSphere.transform.position = pointPos;
		}

		public void DrawAxisLine3D(Color32 color, float from_x, float from_y, float from_z, float to_x, float to_y, float to_z, float r = 2)
		{
			GameObject newLine;
			LineRenderer lRend;
			if (axisLineObjectBufferIndex3D < axisLineObjectBuffer3D.Count)
			{
				newLine = axisLineObjectBuffer3D[axisLineObjectBufferIndex3D];
				lRend = newLine.GetComponent<LineRenderer>();
			}
			else
			{
				newLine = Instantiate(line3D, lines3D.gameObject.transform);
				newLine.layer = line3D.gameObject.layer;
				lRend = newLine.GetComponent<LineRenderer>();
				axisLineObjectBuffer3D.Add(newLine);
			}
			axisLineObjectBufferIndex3D++;
			newLine.SetActive(true);

			Color32 c1 = color;
			c1.a = 128 + 32;

			lRend.startColor = c1;
			lRend.endColor = c1;

			float base_width = r / 2.0f;

			lRend.positionCount = 2;
			lRend.startWidth = base_width;
			lRend.endWidth = base_width;

			//回転させる
			double speed = Math.PI / 100.0f; //1フレームあたりの回転角
			time += Time.deltaTime;
			float tmp_from_x = from_x;
			float tmp_from_y = from_y;
			float tmp_from_z = from_z;
			from_x = (float)(tmp_from_x * Math.Cos(speed / Math.PI * time) + tmp_from_z * Math.Sin(speed / Math.PI * time));
			from_y = tmp_from_y;
			from_z = (float)(-tmp_from_x * Math.Sin(speed / Math.PI * time) + tmp_from_z * Math.Cos(speed / Math.PI * time));
			float tmp_to_x = to_x;
			float tmp_to_y = to_y;
			float tmp_to_z = to_z;
			to_x = (float)(tmp_to_x * Math.Cos(speed / Math.PI * time) + tmp_to_z * Math.Sin(speed / Math.PI * time));
			to_y = tmp_to_y;
			to_z = (float)(-tmp_to_x * Math.Sin(speed / Math.PI * time) + tmp_to_z * Math.Cos(speed / Math.PI * time));

			Vector3 pointPos1 = drawPos; //大きさと位置を調整
			pointPos1.x += from_x * this.scale;
			pointPos1.y += from_y * this.scale;
			pointPos1.z += from_z * this.scale;

			Vector3 pointPos2 = drawPos;
			pointPos2.x += to_x * this.scale;
			pointPos2.y += to_y * this.scale;
			pointPos2.z += to_z * this.scale;

			Vector3 startVec = pointPos1;
			Vector3 endVec = pointPos2;
			lRend.SetPosition(0, startVec);
			lRend.SetPosition(1, endVec);
		}

	}
}



#-- рабочий вар  вывод с нормами и подрядчиком
def sql_quary(proj_id):
    return f'''
WITH table1 AS (
	SELECT 	concat(r.id_activity,'_',CAST(CAST(r.dt as float) as int),'_',id_mat_res) AS "key_connect",
	concat(r.id_activity,'_',CAST(CAST(r.dt as float) as int),'_',id_mat_res,'_',res.id_resource) AS "key_connect2",
	r.id_activity, dt, id_mat_res ,r.id_resource ,r.act_reg_qty ,res.resource_name, res.resource_type
	FROM publication.RESASSIGNMENTSPRED r
	LEFT JOIN publication.RESOURCE res ON r.id_resource = res.id_resource
	LEFT JOIN (
		SELECT id_activity, id_resource AS id_mat_res
		FROM publication.RESASSIGNMENT r
		WHERE resource_type ='RT_Mat') AS res_mat ON res_mat.id_activity =  r.id_activity
	WHERE id_project = {proj_id} AND resource_type ='RT_Equip' and r.act_reg_qty IS NOT NULL and res.resource_name !='Стоимость'),
mat_table AS (
	SELECT 	concat(r.id_activity,'_',CAST(CAST(r.dt as float) as int),'_',r.id_resource) AS "key_connect",
	r.id_resource , r.dt, r.act_reg_qty, res.resource_name
	FROM publication.RESASSIGNMENTSPRED r
	LEFT JOIN publication.RESOURCE res ON r.id_resource = res.id_resource
	WHERE id_project = {proj_id} AND resource_type ='RT_Mat' and r.act_reg_qty IS NOT NULL and res.resource_name !='Стоимость')
	,
contractor AS(
SELECT   concat(r.id_activity,'_',CAST(CAST(r.dt as float) as int),'_',id_mat_res) AS "key_connect",
concat(r.id_activity,'_',CAST(CAST(r.dt as float) as int),'_',id_mat_res,'_',res.id_resource) AS "key_connect2",
  res_mat.contractor_name, r.id_resource , res.resource_name
  FROM publication.RESASSIGNMENTSPRED r
  LEFT JOIN publication.RESOURCE res ON r.id_resource = res.id_resource
  LEFT JOIN (
    SELECT r.id_activity, id_resource AS id_mat_res, uca.contractor_name
    FROM publication.RESASSIGNMENT r
    JOIN publication.UDF_CODE_Activity uca ON r.id_activity  = uca.id_activity
    WHERE resource_type ='RT_Mat') AS res_mat ON res_mat.id_activity =  r.id_activity
  WHERE id_project = {proj_id} AND resource_type ='RT_Equip' and r.act_reg_qty IS NOT NULL and res.resource_name !='Стоимость'
)
SELECT
n.PO_id,n.PO_coef, a.id_activity,
	p.project_name, p.id_project,
	i.isr_name, i.id_ISR,
	a.activity_name, a.id_activity, t1.dt,
	t1.resource_name, t1.id_resource, t1.act_reg_qty,
	mt.resource_name AS "mat_res_name", mt.id_resource AS "mat_id_res", mt.act_reg_qty AS "mat_res_qt"
	,
	contr.contractor_name, contr.id_resource , contr.resource_name
FROM table1 t1
LEFT JOIN mat_table mt ON mt.key_connect = t1.key_connect
LEFT JOIN contractor contr ON  contr.key_connect2 = t1.key_connect2
JOIN publication.ACTIVITY a ON t1.id_activity = a.id_activity
JOIN publication.ISR i ON i.id_ISR = a.id_isr
JOIN publication.PROJECT p ON a.id_project = p.id_project
JOIN (SELECT t.task_id,t.task_name, u.udf_text AS PO_id, u2.udf_text AS PO_coef
  FROM PMDB.privuser.TASK t
  JOIN PMDB.privuser.UDFVALUE u ON u.fk_id = t.task_id
  JOIN PMDB.privuser.UDFVALUE u2 ON u2.fk_id = t.task_id
  WHERE u2.udf_type_id = 7397 AND u.udf_type_id = 3124) AS n ON n.task_id = a.id_activity
WHERE mt.act_reg_qty is not NULL
ORDER BY a.id_activity, t1.dt
'''












